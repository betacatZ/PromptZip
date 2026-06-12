"""
端侧 Reranker + LLM 端到端评估一条命令编排脚本

将 Step 1→2→3→4→5 串联为一条命令，所有参数通过 YAML 配置文件集中管理。

完整流程（数据集切分由 extract_dataset.py 单独执行，不在本 pipeline 中）：
  Step 1: 推送 reranker JSON 到设备 → 运行推理 → 回收分数
  Step 2: parse_and_collect.py → 端侧分数 CSV
  Step 3: device_rerank_compress.py → topk筛选 → 生成LLM推理JSON
  Step 4: 推送 LLM JSON 到设备 → 运行推理 → 回收回答
  Step 5: parse_baseline_output.py → 解析回答 → 评分

使用方式:
    # 先单独执行数据集切分（生成 chunk_output_dir 目录）:
    python extract_dataset.py --mode chunk --save_chunks ...

    # 然后一条命令完成 Step 1→5:
    python run_device_pipeline.py --config ../config/device_pipeline.yaml

    # 从指定步骤开始（前面的步骤已完成）:
    python run_device_pipeline.py --config ../config/device_pipeline.yaml --start_step 4

    # 预览（只打印要执行的命令，不实际运行）:
    python run_device_pipeline.py --config ../config/device_pipeline.yaml --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ============================================================
# 辅助函数：hdc 设备操作
# ============================================================

def hdc_cmd(hdc_addr, args, check=True, verbose=True):
    """
    执行 hdc 命令（通过 zsh -ic 以加载用户 shell profile 中的 PATH）

    Args:
        hdc_addr: hdc 连接地址 (如 100.103.109.221:8710)
        args: hdc 命令参数列表 (如 ["shell", "mkdir -p /data/test"])
        check: 是否检查返回码
        verbose: 是否打印命令
    """
    cmd_str = subcmd_str(["hdc", "-s", hdc_addr] + args)
    if verbose:
        print(f"  [hdc] {cmd_str}")
    result = subprocess.run(["zsh", "-ic", cmd_str], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        print(f"  [hdc] 命令失败: {result.stderr}")
        raise RuntimeError(f"hdc 命令失败: {cmd_str}\nstderr: {result.stderr}")
    return result


def subcmd_str(cmd):
    """将命令列表转为可读字符串"""
    return " ".join(cmd)


def hdc_mkdir(hdc_addr, device_dir):
    """在设备上创建目录"""
    return hdc_cmd(hdc_addr, ["shell", f"mkdir -p {device_dir}"])


def hdc_file_send(hdc_addr, local_path, device_path):
    """发送文件到设备"""
    return hdc_cmd(hdc_addr, ["file", "send", local_path, device_path])


def hdc_file_recv(hdc_addr, device_path, local_path):
    """从设备回收文件"""
    return hdc_cmd(hdc_addr, ["file", "recv", device_path, local_path])


def hdc_shell(hdc_addr, command):
    """在设备上执行 shell 命令"""
    return hdc_cmd(hdc_addr, ["shell", command])


def hdc_cleanup(hdc_addr, device_dirs_patterns):
    """
    清理设备上的文件

    Args:
        hdc_addr: hdc 连接地址
        device_dirs_patterns: 列表，每个元素是 (device_dir, glob_pattern)
            如 [("/data/qwen3/test", "*.json"), ("/data/qwen3/output", "*.txt")]
    """
    for device_dir, pattern in device_dirs_patterns:
        hdc_cmd(hdc_addr, ["shell", f"rm -f {device_dir}/{pattern}"],
                check=False, verbose=False)


# ============================================================
# 辅助函数：运行 Python 子脚本
# ============================================================

def run_python_script(script_path, args_dict, verbose=True, check=True):
    """
    运行 Python 脚本，传递 argparse 参数

    Args:
        script_path: Python 脚本路径
        args_dict: 参数字典，如 {"input_dir": "/path", "output_file": "/path.csv"}
        verbose: 是否打印命令
        check: 是否检查返回码
    """
    cmd = [sys.executable, script_path]
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif value is not None:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    if verbose:
        print(f"  [python] {subcmd_str(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.stdout:
        # 打印子脚本输出（但限制长度避免刷屏）
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 30:
            for line in lines:
                print(f"    {line}")
        else:
            for line in lines[:15]:
                print(f"    {line}")
            print(f"    ... (省略 {len(lines) - 30} 行)")
            for line in lines[-15:]:
                print(f"    {line}")
    if check and result.returncode != 0:
        print(f"  [python] 脚本失败 (returncode={result.returncode})")
        if result.stderr:
            print(f"    stderr: {result.stderr[:500]}")
        raise RuntimeError(f"Python 脚本失败: {subcmd_str(cmd)}")
    return result


# ============================================================
# Pipeline 核心类
# ============================================================

# 步骤编号映射
STEP_ORDER = ["1", "2", "3", "4", "5"]


class DevicePipeline:
    def __init__(self, config_path, start_step=None, dry_run=False):
        self.config_path = config_path
        self.dry_run = dry_run
        self.start_step = start_step

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 推导数据集名称
        self.dataset_name = self.config.get("dataset_name")
        if not self.dataset_name:
            # 从 input_file basename 推导
            input_file = self.config["input_file"]
            self.dataset_name = Path(input_file).stem  # 如 narrativeqa.jsonl → narrativeqa

        # 推导各步骤路径
        self.paths = self._derive_paths()

        # 获取脚本目录（pipeline 所在目录）
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def _derive_paths(self):
        """
        基于 output_root 和 dataset_name 自动推导各步骤的输入输出路径

        目录结构:
          output_root/{dataset_name}/
            device_rerank_output/    ← Step 1 输出 (reranker 分数 .txt)
            scores.csv               ← Step 2 输出
            llm_json_output/         ← Step 3 输出 (LLM推理JSON)
            compressed_info/         ← Step 3 可选输出
            device_llm_output/       ← Step 4 输出 (LLM回答 .txt)
            eval_results/            ← Step 5 输出
        """
        root = self.config["output_root"]
        ds = self.dataset_name
        base = os.path.join(root, ds)

        paths = {
            # 输入：chunk_output_dir 来自配置（Step 1 的输出）
            "chunk_output_dir": self.config["chunk_output_dir"],
            "chunk_inputs_dir": os.path.join(self.config["chunk_output_dir"], "inputs"),
            "input_file": self.config["input_file"],
            "dataset_dir": self.config["dataset_dir"],
            # Step 1 输出
            "device_rerank_output_dir": os.path.join(base, "device_rerank_output"),
            # Step 2 输出
            "scores_csv": os.path.join(base, "scores.csv"),
            # Step 3 输出
            "llm_json_output_dir": os.path.join(base, "llm_json_output"),
            # Step 4 输出
            "device_llm_output_dir": os.path.join(base, "device_llm_output"),
            # Step 5 输出
            "eval_results_dir": os.path.join(base, "eval_results"),
            # base
            "base_dir": base,
        }
        return paths

    def _should_run(self, step_num):
        """判断某个步骤是否应该执行"""
        if self.start_step is None:
            return True
        # 比较步骤编号
        step_idx = STEP_ORDER.index(step_num)
        start_idx = STEP_ORDER.index(self.start_step)
        return step_idx >= start_idx

    def _ensure_dir(self, path):
        """确保目录存在"""
        if not self.dry_run:
            os.makedirs(path, exist_ok=True)
        else:
            print(f"  [dry_run] mkdir -p {path}")

    # ============================================================
    # Step 1: 推送 reranker JSON → 设备推理 → 回收分数
    # ============================================================

    def run_step1(self):
        """替代 prepare_and_run.sh，所有参数从 YAML 配置读取"""
        print("\n" + "=" * 60)
        print("Step 1: 端侧 Reranker 推理")
        print("=" * 60)

        cfg = self.config["reranker_device"]
        hdc_addr = cfg["hdc_addr"]
        device_test_dir = cfg["device_test_dir"]
        device_output_dir = cfg["device_output_dir"]
        qwen_path = cfg["qwen_path"]
        ld_library_path = cfg["ld_library_path"]

        local_input_dir = self.paths["chunk_output_dir"]
        local_output_dir = self.paths["device_rerank_output_dir"]

        if self.dry_run:
            print(f"  [dry_run] hdc -s {hdc_addr} shell mkdir -p {device_test_dir}")
            print(f"  [dry_run] hdc -s {hdc_addr} shell mkdir -p {device_output_dir}")
            print(f"  [dry_run] hdc -s {hdc_addr} file send {local_input_dir}/*.json {device_test_dir}/")
            print(f"  [dry_run] hdc -s {hdc_addr} shell <推理脚本>")
            print(f"  [dry_run] hdc -s {hdc_addr} file recv {device_output_dir}/*.txt {local_output_dir}/")
            print(f"  [dry_run] hdc -s {hdc_addr} shell rm -f {device_test_dir}/*.json {device_output_dir}/*.txt")
            return

        # 检查本地输入目录
        if not os.path.isdir(local_input_dir):
            raise RuntimeError(f"本地输入目录不存在: {local_input_dir}\n请先执行 extract_dataset.py --mode chunk --save_chunks")

        # 准备本地输出目录
        self._ensure_dir(local_output_dir)

        # 统计 JSON 文件
        json_files = sorted(Path(local_input_dir).glob("*.json"))
        # 排除 inputs/ 下的 chunks.json（它们不在根目录）
        json_files = [f for f in json_files if f.parent == Path(local_input_dir)]
        if not json_files:
            raise RuntimeError(f"在 {local_input_dir} 中没有找到 JSON 文件")

        print(f"  共 {len(json_files)} 个 JSON 文件")

        # 1. 在设备上创建目录
        hdc_mkdir(hdc_addr, device_test_dir)
        hdc_mkdir(hdc_addr, device_output_dir)

        # 2. 发送 JSON 文件到设备
        print("  发送 JSON 文件到设备...")
        for i, json_file in enumerate(json_files):
            filename = json_file.name
            print(f"    [{i+1}/{len(json_files)}] 发送: {filename}")
            hdc_file_send(hdc_addr, str(json_file), f"{device_test_dir}/{filename}")

        # 3. 在设备上运行推理
        print("  在设备上运行推理...")
        # 生成推理脚本
        script_lines = [
            "set -e",
            f"export LD_LIBRARY_PATH={ld_library_path}",
            f"QWEN3_ABS_PATH=\"{qwen_path}\"",
            "total=0",
            f"for f in {device_test_dir}/*.json; do",
            "    if [ -f \"$f\" ]; then total=$((total + 1)); fi",
            "done",
            "current=0",
            f"for json_file in {device_test_dir}/*.json; do",
            "    if [ -f \"$json_file\" ]; then",
            "        current=$((current + 1))",
            "        filename=$(basename \"$json_file\")",
            "        stem=$(basename \"$json_file\" .json)",
            f"        output_file=\"{device_output_dir}/$stem.txt\"",
            "        echo \"[$current/$total] 处理: $filename\"",
            "        \"$QWEN3_ABS_PATH\" \"$json_file\" > \"$output_file\" 2>&1 || {",
            "            echo \"  警告: $filename 处理失败\"",
            "        }",
            "    fi",
            "done",
            "echo '推理完成'",
        ]

        # 创建临时脚本，发送到设备，执行
        temp_script = None
        try:
            import tempfile
            temp_script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
            temp_script.write("\n".join(script_lines))
            temp_script.close()

            hdc_file_send(hdc_addr, temp_script.name, "/data/qwen3/qwen_script.sh")
            hdc_shell(hdc_addr, "chmod +x /data/qwen3/qwen_script.sh && /data/qwen3/qwen_script.sh")
        finally:
            if temp_script:
                os.unlink(temp_script.name)
            hdc_cmd(hdc_addr, ["shell", "rm -f /data/qwen3/qwen_script.sh"],
                    check=False, verbose=False)

        # 4. 回收输出文件
        print("  回收输出文件...")
        recv_count = 0
        for i, json_file in enumerate(json_files):
            txt_name = json_file.stem + ".txt"
            device_path = f"{device_output_dir}/{txt_name}"
            local_path = os.path.join(local_output_dir, txt_name)
            print(f"    [{i+1}/{len(json_files)}] 回收: {txt_name}")
            hdc_file_recv(hdc_addr, device_path, local_path)
            recv_count += 1

        print(f"  ✅ 回收了 {recv_count} 个文件")

        # 5. 清理设备上的测试文件
        print("  清理设备上的测试文件...")
        hdc_cleanup(hdc_addr, [
            (device_test_dir, "*.json"),
            (device_output_dir, "*.txt"),
        ])

        # 验证输出
        output_txts = list(Path(local_output_dir).glob("*.txt"))
        print(f"  验证: 本地输出目录有 {len(output_txts)} 个 .txt 文件")
        if len(output_txts) == 0:
            raise RuntimeError("Step 1 输出验证失败: 没有 .txt 文件被回收")

    # ============================================================
    # Step 2: parse_and_collect.py
    # ============================================================

    def run_step2(self):
        """从端侧输出提取 correlation coefficient → CSV"""
        print("\n" + "=" * 60)
        print("Step 2: 收集端侧分数 (parse_and_collect)")
        print("=" * 60)

        input_dir = self.paths["device_rerank_output_dir"]
        output_file = self.paths["scores_csv"]

        script_path = os.path.join(self.script_dir, "parse_and_collect.py")
        args = {
            "input_dir": input_dir,
            "output_file": output_file,
        }

        if self.dry_run:
            print(f"  [dry_run] python {script_path} --input_dir {input_dir} --output_file {output_file}")
            return

        # 检查输入目录
        if not os.path.isdir(input_dir):
            raise RuntimeError(f"Step 2 输入目录不存在: {input_dir}")

        result = run_python_script(script_path, args)

        # 验证输出
        if not os.path.isfile(output_file):
            raise RuntimeError(f"Step 2 输出验证失败: CSV 文件不存在: {output_file}")

        # 打印 CSV 行数
        with open(output_file, "r") as f:
            lines = f.readlines()
        print(f"  ✅ scores.csv 共 {len(lines) - 1} 行数据（不含表头）")

    # ============================================================
    # Step 3: device_rerank_compress.py
    # ============================================================

    def run_step3(self):
        """读取 chunks + 端侧分数 → topk筛选 → 生成LLM推理JSON"""
        print("\n" + "=" * 60)
        print("Step 3: 端侧分数 → topk筛选 → 生成LLM推理JSON")
        print("=" * 60)

        cfg = self.config["compress"]
        chunks_dir = self.paths["chunk_inputs_dir"]
        scores_csv = self.paths["scores_csv"]
        input_file = self.paths["input_file"]
        output_dir = self.paths["llm_json_output_dir"]

        script_path = os.path.join(self.script_dir, "device_rerank_compress.py")
        args = {
            "chunks_dir": chunks_dir,
            "scores_csv": scores_csv,
            "input_file": input_file,
            "output_dir": output_dir,
            "rate": cfg["rate"],
            "max_ctx": cfg["max_ctx"],
            "tokenizer_path": cfg["tokenizer_path"],
            "params_path": cfg["params_path"],
            "local_tokenizer_path": cfg.get("local_tokenizer_path"),
            "model_name": cfg.get("model_name"),
        }
        if cfg.get("save_compressed"):
            args["save_compressed"] = True

        if self.dry_run:
            parts = [f"python {script_path}"]
            for k, v in args.items():
                if isinstance(v, bool) and v:
                    parts.append(f"--{k}")
                elif v is not None:
                    parts.append(f"--{k} {v}")
            print(f"  [dry_run] {subcmd_str(parts)}")
            return

        # 检查输入
        if not os.path.isdir(chunks_dir):
            raise RuntimeError(f"chunks 目录不存在: {chunks_dir}")
        if not os.path.isfile(scores_csv):
            raise RuntimeError(f"scores CSV 不存在: {scores_csv}")

        self._ensure_dir(output_dir)

        result = run_python_script(script_path, args)

        # 验证输出
        json_files = list(Path(output_dir).glob("*.json"))
        # 排除 compressed_info 子目录中的文件
        json_files = [f for f in json_files if f.parent == Path(output_dir)]
        print(f"  ✅ 生成了 {len(json_files)} 个 LLM 推理 JSON 文件")
        if len(json_files) == 0:
            raise RuntimeError("Step 3 输出验证失败: 没有 JSON 文件")

    # ============================================================
    # Step 4: 推送 LLM JSON → 设备推理 → 回收回答
    # ============================================================

    def run_step4(self):
        """替代 run_baseline_device.sh，所有参数从 YAML 配置读取"""
        print("\n" + "=" * 60)
        print("Step 4: 端侧 LLM 推理")
        print("=" * 60)

        cfg = self.config["llm_device"]
        hdc_addr = cfg["hdc_addr"]
        device_qwen_dir = cfg["device_qwen_dir"]
        device_test_dir = cfg["device_test_dir"]
        device_output_dir = cfg["device_output_dir"]

        local_input_dir = self.paths["llm_json_output_dir"]
        local_output_dir = self.paths["device_llm_output_dir"]

        if self.dry_run:
            print(f"  [dry_run] hdc -s {hdc_addr} shell mkdir -p {device_test_dir}")
            print(f"  [dry_run] hdc -s {hdc_addr} shell mkdir -p {device_output_dir}")
            print(f"  [dry_run] hdc -s {hdc_addr} file send {local_input_dir}/*.json {device_test_dir}/")
            print(f"  [dry_run] hdc -s {hdc_addr} shell <推理脚本>")
            print(f"  [dry_run] hdc -s {hdc_addr} file recv {device_output_dir}/*.txt {local_output_dir}/")
            return

        # 检查本地输入目录
        if not os.path.isdir(local_input_dir):
            raise RuntimeError(f"本地输入目录不存在: {local_input_dir}")

        # 准备本地输出目录
        self._ensure_dir(local_output_dir)

        # 统计 JSON 文件
        json_files = sorted(Path(local_input_dir).glob("*.json"))
        # 排除子目录中的文件
        json_files = [f for f in json_files if f.parent == Path(local_input_dir)]
        if not json_files:
            raise RuntimeError(f"在 {local_input_dir} 中没有找到 JSON 文件")

        print(f"  共 {len(json_files)} 个 JSON 文件")

        # 1. 在设备上创建目录
        hdc_mkdir(hdc_addr, device_test_dir)
        hdc_mkdir(hdc_addr, device_output_dir)

        # 2. 发送 JSON 文件到设备
        print("  发送 JSON 文件到设备...")
        for i, json_file in enumerate(json_files):
            filename = json_file.name
            print(f"    [{i+1}/{len(json_files)}] 发送: {filename}")
            hdc_file_send(hdc_addr, str(json_file), f"{device_test_dir}/{filename}")

        # 3. 在设备上运行推理
        print("  在设备上运行推理...")
        script_lines = [
            "set -e",
            "export LD_LIBRARY_PATH=./",
            f"cd {device_qwen_dir}",
            "total=0",
            f"for f in {device_test_dir}/*.json; do",
            "    if [ -f \"$f\" ]; then total=$((total + 1)); fi",
            "done",
            "current=0",
            f"for json_file in {device_test_dir}/*.json; do",
            "    if [ -f \"$json_file\" ]; then",
            "        current=$((current + 1))",
            "        filename=$(basename \"$json_file\")",
            "        stem=$(basename \"$json_file\" .json)",
            f"        output_file=\"{device_output_dir}/$stem.txt\"",
            "        echo \"[$current/$total] 处理: $filename\"",
            f"        {device_qwen_dir}/qwen2 \"$json_file\" > \"$output_file\" 2>&1 || {{",
            "            echo \"  警告: $filename 处理失败\"",
            "        }}",
            "    fi",
            "done",
            "echo '推理完成'",
        ]

        temp_script = None
        try:
            import tempfile
            temp_script = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
            temp_script.write("\n".join(script_lines))
            temp_script.close()

            hdc_file_send(hdc_addr, temp_script.name, f"{device_qwen_dir}/baseline_script.sh")
            hdc_shell(hdc_addr, f"chmod +x {device_qwen_dir}/baseline_script.sh && {device_qwen_dir}/baseline_script.sh")
        finally:
            if temp_script:
                os.unlink(temp_script.name)
            hdc_cmd(hdc_addr, ["shell", f"rm -f {device_qwen_dir}/baseline_script.sh"],
                    check=False, verbose=False)

        # 4. 回收输出文件
        print("  回收输出文件...")
        recv_count = 0
        for i, json_file in enumerate(json_files):
            txt_name = json_file.stem + ".txt"
            device_path = f"{device_output_dir}/{txt_name}"
            local_path = os.path.join(local_output_dir, txt_name)
            print(f"    [{i+1}/{len(json_files)}] 回收: {txt_name}")
            hdc_file_recv(hdc_addr, device_path, local_path)
            recv_count += 1

        print(f"  ✅ 回收了 {recv_count} 个文件")

        # 5. 清理设备上的测试文件
        print("  清理设备上的测试文件...")
        hdc_cleanup(hdc_addr, [
            (device_test_dir, "*.json"),
            (device_output_dir, "*.txt"),
        ])

        # 验证输出
        output_txts = list(Path(local_output_dir).glob("*.txt"))
        print(f"  验证: 本地输出目录有 {len(output_txts)} 个 .txt 文件")
        if len(output_txts) == 0:
            raise RuntimeError("Step 4 输出验证失败: 没有 .txt 文件被回收")

    # ============================================================
    # Step 5: parse_baseline_output.py
    # ============================================================

    def run_step5(self):
        """解析端侧 LLM 输出 → 评分"""
        print("\n" + "=" * 60)
        print("Step 5: 解析端侧输出 & 评分")
        print("=" * 60)

        input_dir = self.paths["device_llm_output_dir"]
        dataset_dir = self.paths["dataset_dir"]
        output_dir = self.paths["eval_results_dir"]

        script_path = os.path.join(self.script_dir, "parse_baseline_output.py")
        args = {
            "input_dir": input_dir,
            "dataset_dir": dataset_dir,
            "output_dir": output_dir,
        }

        if self.dry_run:
            parts = [f"python {script_path}"]
            for k, v in args.items():
                parts.append(f"--{k} {v}")
            print(f"  [dry_run] {subcmd_str(parts)}")
            return

        # 检查输入
        if not os.path.isdir(input_dir):
            raise RuntimeError(f"Step 5 输入目录不存在: {input_dir}")

        self._ensure_dir(output_dir)

        result = run_python_script(script_path, args)

        # 验证输出
        score_json = os.path.join(output_dir, "score.json")
        if os.path.isfile(score_json):
            with open(score_json, "r") as f:
                scores = json.load(f)
            print(f"\n  ✅ 评分结果:")
            for task, info in scores.items():
                if isinstance(info, dict):
                    print(f"    {task}: score={info['score']}, num={info['num']}")
                else:
                    print(f"    {task}: {info}")
        else:
            raise RuntimeError(f"Step 5 输出验证失败: score.json 不存在于 {output_dir}")

    # ============================================================
    # 主运行方法
    # ============================================================

    def run(self):
        """按顺序执行所有步骤"""
        print("=" * 60)
        print("端侧 Reranker + LLM 端到端评估 Pipeline")
        print("=" * 60)
        print(f"配置文件: {self.config_path}")
        print(f"数据集: {self.dataset_name}")
        print(f"输出根目录: {self.paths['base_dir']}")
        if self.dry_run:
            print("⚠️  DRY RUN 模式 — 仅打印命令，不实际执行")
        if self.start_step:
            print(f"从 Step {self.start_step} 开始")
        print()

        steps = {
            "1": ("端侧 Reranker 推理", self.run_step1),
            "2": ("收集端侧分数", self.run_step2),
            "3": ("topk筛选 → 生成LLM推理JSON", self.run_step3),
            "4": ("端侧 LLM 推理", self.run_step4),
            "5": ("解析输出 & 评分", self.run_step5),
        }

        total_time = 0
        completed = []

        for step_num in STEP_ORDER:
            if not self._should_run(step_num):
                print(f"\n⏭️  跳过 Step {step_num}: {steps[step_num][0]}")
                continue

            desc, func = steps[step_num]
            print(f"\n▶️  Step {step_num}: {desc}")

            start_time = time.time()
            try:
                func()
                elapsed = time.time() - start_time
                total_time += elapsed
                completed.append(step_num)
                print(f"  ⏱️  耗时: {elapsed:.1f}s")
            except RuntimeError as e:
                elapsed = time.time() - start_time
                print(f"\n❌ Step {step_num} 失败: {e}")
                print(f"  ⏱️  耗时: {elapsed:.1f}s")
                print(f"\n💡 你可以从失败的步骤重新开始:")
                print(f"  python run_device_pipeline.py --config {self.config_path} --start_step {step_num}")
                sys.exit(1)

        # 最终总结
        print("\n" + "=" * 60)
        print("Pipeline 完成!")
        print("=" * 60)
        print(f"  数据集: {self.dataset_name}")
        print(f"  完成步骤: {completed}")
        print(f"  总耗时: {total_time:.1f}s")
        print(f"\n  输出目录结构:")
        for name, path in self.paths.items():
            if os.path.exists(path):
                print(f"    {name}: {path}")

        # 如果 Step 5 完成，打印评分摘要
        score_json = os.path.join(self.paths["eval_results_dir"], "score.json")
        if os.path.isfile(score_json):
            print(f"\n  📊 评分结果: {score_json}")

        print()


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="端侧 Reranker + LLM 端到端评估一条命令编排脚本\n\n"
                    "完整流程: Step 1→2→3→4→5\n"
                    "  Step 1: 端侧 Reranker 推理\n"
                    "  Step 2: 收集端侧分数\n"
                    "  Step 3: topk筛选 → 生成LLM推理JSON\n"
                    "  Step 4: 端侧 LLM 推理\n"
                    "  Step 5: 解析输出 & 评分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径 (如 ../config/device_pipeline.yaml)",
    )
    parser.add_argument(
        "--start_step",
        type=str,
        default=None,
        choices=STEP_ORDER,
        help="从指定步骤开始执行 (前面的步骤假定已完成)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="预览模式：只打印要执行的命令，不实际运行",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="覆盖 YAML 中的 input_file (原始数据集 JSONL)",
    )
    parser.add_argument(
        "--chunk_output_dir",
        type=str,
        default=None,
        help="覆盖 YAML 中的 chunk_output_dir (Step 1 的输出目录)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="覆盖 YAML 中的 output_root",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="覆盖 YAML 中的 compress.rate (压缩率)",
    )
    parser.add_argument(
        "--max_ctx",
        type=int,
        default=None,
        help="覆盖 YAML 中的 compress.max_ctx",
    )

    args = parser.parse_args()

    # 创建 pipeline 实例
    pipeline = DevicePipeline(
        config_path=args.config,
        start_step=args.start_step,
        dry_run=args.dry_run,
    )

    # CLI 参数覆盖 YAML 配置
    if args.input_file:
        pipeline.config["input_file"] = args.input_file
        pipeline.dataset_name = Path(args.input_file).stem
        pipeline.paths = pipeline._derive_paths()
    if args.chunk_output_dir:
        pipeline.config["chunk_output_dir"] = args.chunk_output_dir
        pipeline.paths = pipeline._derive_paths()
    if args.output_root:
        pipeline.config["output_root"] = args.output_root
        pipeline.paths = pipeline._derive_paths()
    if args.rate:
        pipeline.config["compress"]["rate"] = args.rate
    if args.max_ctx:
        pipeline.config["compress"]["max_ctx"] = args.max_ctx

    # 运行 pipeline
    pipeline.run()


if __name__ == "__main__":
    main()