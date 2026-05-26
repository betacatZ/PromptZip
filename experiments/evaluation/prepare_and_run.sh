
#!/usr/bin/env bash
# 兼容 zsh 和 bash

# ================================================
# 端侧推理自动化脚本
# 功能：
# 1. 发送测试用的 JSON 文件到设备
# 2. 在设备上运行推理
# 3. 回收输出文件
# ================================================

set -e

# 设置 shell 兼容性选项（对于 zsh）
if [ -n "$ZSH_VERSION" ]; then
    setopt NO_NOMATCH  # 在无匹配文件时不报错
fi

# 配置（请根据实际情况修改）
LOCAL_INPUT_DIR="./outputs"              # 本地包含 JSON 文件的目录
DEVICE_TEST_DIR="/data/zhangdeming/test" # 设备上的测试目录
DEVICE_OUTPUT_DIR="/data/zhangdeming/output" # 设备上的输出目录
LOCAL_OUTPUT_DIR="./device_outputs"       # 本地回收文件的目录
QWEN3_PATH="/data/qwen3/qwen3"           # 设备上 qwen3 可执行文件的绝对路径
TOKENIZER_PATH="/data/qwen3/qwen3-reranker-0.6b/Q4_N_0_G128/tokenizer.json" # 设备上 tokenizer 路径
PARAMS_PATH="/data/qwen3/qwen3-reranker-0.6b/Q4_N_0_G128/params" # 设备上参数路径

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -i, --input DIR    本地包含 JSON 文件的目录（默认: $LOCAL_INPUT_DIR）"
    echo "  -o, --output DIR   本地回收文件的目录（默认: $LOCAL_OUTPUT_DIR）"
    echo "  -h, --help         显示帮助信息"
}

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input) LOCAL_INPUT_DIR="$2"; shift ;;
        -o|--output) LOCAL_OUTPUT_DIR="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "未知参数: $1"; show_help; exit 1 ;;
    esac
    shift
done

echo "========================================"
echo "端侧推理自动化脚本"
echo "========================================"
echo "本地输入目录: $LOCAL_INPUT_DIR"
echo "本地输出目录: $LOCAL_OUTPUT_DIR"
echo "设备上的 qwen3 路径: $QWEN3_PATH"
echo ""

# 1. 检查本地输入目录
if [ ! -d "$LOCAL_INPUT_DIR" ]; then
    echo "错误: 本地输入目录不存在: $LOCAL_INPUT_DIR"
    exit 1
fi

# 2. 准备本地输出目录
mkdir -p "$LOCAL_OUTPUT_DIR"

# 3. 在设备上创建目录
echo "在设备上创建目录..."
shhdc shell "mkdir -p $DEVICE_TEST_DIR && mkdir -p $DEVICE_OUTPUT_DIR"

# 4. 发送所有 JSON 文件到设备（只在当前目录查找）
echo "发送 JSON 文件到设备..."
# 兼容 zsh 和 bash 的方式检查和遍历
has_json_files=0
for json_file in "$LOCAL_INPUT_DIR"/*.json; do
    # 检查是否真的存在文件（处理通配符没有匹配的情况）
    if [ -e "$json_file" ]; then
        filename=$(basename "$json_file")
        echo "  发送: $filename"
        hdcsend "$json_file" "$DEVICE_TEST_DIR/$filename"
        has_json_files=1
    fi
done

if [ "$has_json_files" -eq 0 ]; then
    echo "  警告: 在 $LOCAL_INPUT_DIR 中没有找到 JSON 文件"
fi

# 5. 在设备上运行推理
echo ""
echo "在设备上运行推理..."
shhdc shell << EOF
    set -e
    
    # 确保使用绝对路径
    QWEN3_ABS_PATH="$QWEN3_PATH"
    
    # 遍历所有测试 JSON 文件
    for json_file in $DEVICE_TEST_DIR/*.json; do
        if [ -f "\$json_file" ]; then
            filename=\$(basename "\$json_file" .json)
            output_file="$DEVICE_OUTPUT_DIR/\$filename.txt"
            
            echo "正在处理: \$filename"
            echo "  输入文件: \$json_file"
            echo "  输出文件: \$output_file"
            
            # 使用绝对路径执行
            "\$QWEN3_ABS_PATH" "\$json_file" > "\$output_file" 2>&1 || {
                echo "  警告: \$filename 处理失败"
            }
        fi
    done

    echo "推理完成"
EOF

# 6. 回收输出文件（只在当前目录查找）
echo ""
echo "回收输出文件..."
# 兼容 zsh 和 bash 的方式检查和遍历
has_json_files=0
for json_file in "$LOCAL_INPUT_DIR"/*.json; do
    # 检查是否真的存在文件（处理通配符没有匹配的情况）
    if [ -e "$json_file" ]; then
        filename=$(basename "$json_file" .json)
        output_file="$filename.txt"
        device_path="$DEVICE_OUTPUT_DIR/$output_file"
        local_path="$LOCAL_OUTPUT_DIR/$output_file"
        
        echo "  回收: $output_file"
        hdcrecv "$device_path" "$local_path"
        has_json_files=1
    fi
done

if [ "$has_json_files" -eq 0 ]; then
    echo "  警告: 在 $LOCAL_INPUT_DIR 中没有找到 JSON 文件"
fi

# 7. 清理设备上的测试文件（可选）
read -p "是否清理设备上的测试文件？(y/N): " clean_confirm
if [[ "$clean_confirm" =~ ^[Yy]$ ]]; then
    echo "清理设备上的测试文件..."
    shhdc shell "rm -f $DEVICE_TEST_DIR/*.json && rm -f $DEVICE_OUTPUT_DIR/*.txt"
fi

echo ""
echo "========================================"
echo "处理完成!"
echo "输出文件已回收至: $LOCAL_OUTPUT_DIR"
echo "下一步请运行: python parse_and_collect.py --input_dir $LOCAL_OUTPUT_DIR"
echo "========================================"

