source ~/.zshrc
# 兼容 zsh 和 bash

# ================================================
# Baseline端侧推理自动化脚本（Qwen2.5-7b-instruct）
# 功能：
# 1. 发送 baseline JSON 文件到设备
# 2. 在设备上逐条运行 qwen 推理
# 3. 回收输出文件
#
# 单条命令格式：
#   cd /data/qwen
#   export LD_LIBRARY_PATH=./
#   ./qwen ./qwen.json
# ================================================

set -e

# 设置 shell 兼容性选项（对于 zsh）
if [ -n "$ZSH_VERSION" ]; then
    setopt NO_NOMATCH  # 在无匹配文件时不报错
fi

# 配置（请根据实际情况修改）
LOCAL_INPUT_DIR="/data8/zhangdeming/PromptZip/GEWU-dataset-baseline/test"      # 本地包含 baseline JSON 文件的目录
DEVICE_QWEN_DIR="/data/qwen2"                                          # 设备上 qwen 工作目录
DEVICE_TEST_DIR="/data/qwen2/test"                                      # 设备上的测试JSON目录
DEVICE_OUTPUT_DIR="/data/qwen2/output/test"                             # 设备上的输出目录
LOCAL_OUTPUT_DIR="/data8/zhangdeming/PromptZip/GEWU_output/baseline/test" # 本地回收文件的目录
HDC_ADDR="100.103.109.221:8710"                                         # hdc 连接地址

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -i, --input DIR    本地包含 baseline JSON 文件的目录（默认: $LOCAL_INPUT_DIR）"
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
echo "Baseline 端侧推理自动化脚本"
echo "========================================"
echo "本地输入目录: $LOCAL_INPUT_DIR"
echo "本地输出目录: $LOCAL_OUTPUT_DIR"
echo "设备上 qwen 目录: $DEVICE_QWEN_DIR"
echo ""

# 1. 检查本地输入目录
if [ ! -d "$LOCAL_INPUT_DIR" ]; then
    echo "错误: 本地输入目录不存在: $LOCAL_INPUT_DIR"
    exit 1
fi

# 2. 准备本地输出目录
mkdir -p "$LOCAL_OUTPUT_DIR"

# 3. 在设备上创建输出目录
echo "在设备上创建目录..."
hdc -s "$HDC_ADDR" shell "mkdir -p $DEVICE_TEST_DIR && mkdir -p $DEVICE_OUTPUT_DIR"

# 4. 发送所有 baseline JSON 文件到设备的测试目录
echo "发送 JSON 文件到设备..."
has_json_files=0
for json_file in "$LOCAL_INPUT_DIR"/*.json; do
    if [ -e "$json_file" ]; then
        filename=$(basename "$json_file")
        echo "  发送: $filename"
        hdc -s "$HDC_ADDR" file send "$json_file" "$DEVICE_TEST_DIR/$filename"
        has_json_files=1
    fi
done

if [ "$has_json_files" -eq 0 ]; then
    echo "  警告: 在 $LOCAL_INPUT_DIR 中没有找到 JSON 文件"
    exit 0
fi

# 5. 在设备上逐条运行推理
# baseline模式：每次将JSON重命名为 qwen.json，cd到qwen目录，
# 设置LD_LIBRARY_PATH=./，执行 ./qwen ./qwen.json
echo ""
echo "在设备上运行推理..."

# 创建临时脚本文件
TEMP_SCRIPT=$(mktemp /tmp/qwen_baseline_script.XXXXXX)

printf "%s\n" "set -e" > "$TEMP_SCRIPT"
printf "%s\n" "" >> "$TEMP_SCRIPT"
printf "%s\n" "# 设置库路径" >> "$TEMP_SCRIPT"
printf "%s\n" "export LD_LIBRARY_PATH=./" >> "$TEMP_SCRIPT"
printf "%s\n" "" >> "$TEMP_SCRIPT"
printf "%s\n" "cd $DEVICE_QWEN_DIR" >> "$TEMP_SCRIPT"
printf "%s\n" "" >> "$TEMP_SCRIPT"
printf "%s\n" "# 遍历所有 baseline JSON 文件" >> "$TEMP_SCRIPT"
printf "%s\n" "for json_file in $DEVICE_TEST_DIR/*.json; do" >> "$TEMP_SCRIPT"
printf "%s\n" "    if [ -f \"\$json_file\" ]; then" >> "$TEMP_SCRIPT"
printf "%s\n" "        filename=\$(basename \"\$json_file\")" >> "$TEMP_SCRIPT"
printf "%s\n" "        stem=\$(basename \"\$json_file\" .json)" >> "$TEMP_SCRIPT"
printf "%s\n" "        output_file=\"$DEVICE_OUTPUT_DIR/\$stem.txt\"" >> "$TEMP_SCRIPT"
printf "%s\n" "        " >> "$TEMP_SCRIPT"
printf "%s\n" "        echo \"正在处理: \$filename\"" >> "$TEMP_SCRIPT"
printf "%s\n" "        " >> "$TEMP_SCRIPT"
printf "%s\n" "        # 执行推理，直接使用绝对路径" >> "$TEMP_SCRIPT"
printf "%s\n" "        $DEVICE_QWEN_DIR/qwen2 \"\$json_file\" > \"\$output_file\" 2>&1 || {" >> "$TEMP_SCRIPT"
printf "%s\n" "            echo \"  警告: \$filename 处理失败\"" >> "$TEMP_SCRIPT"
printf "%s\n" "        }" >> "$TEMP_SCRIPT"
printf "%s\n" "    fi" >> "$TEMP_SCRIPT"
printf "%s\n" "done" >> "$TEMP_SCRIPT"
printf "%s\n" "" >> "$TEMP_SCRIPT"
printf "%s\n" "echo \"推理完成\"" >> "$TEMP_SCRIPT"

# 发送脚本到设备
hdc -s "$HDC_ADDR" file send "$TEMP_SCRIPT" $DEVICE_QWEN_DIR/baseline_script.sh

# 在设备上执行脚本
hdc -s "$HDC_ADDR" shell "chmod +x $DEVICE_QWEN_DIR/baseline_script.sh && $DEVICE_QWEN_DIR/baseline_script.sh"

# 清理
rm -f "$TEMP_SCRIPT"
hdc -s "$HDC_ADDR" shell "rm -f $DEVICE_QWEN_DIR/baseline_script.sh" 2>/dev/null || true

# 6. 回收输出文件
echo ""
echo "回收输出文件..."
has_json_files=0
for json_file in "$LOCAL_INPUT_DIR"/*.json; do
    if [ -e "$json_file" ]; then
        filename=$(basename "$json_file" .json)
        output_file="$filename.txt"
        device_path="$DEVICE_OUTPUT_DIR/$output_file"
        local_path="$LOCAL_OUTPUT_DIR/$output_file"

        echo "  回收: $output_file"
        hdc -s "$HDC_ADDR" file recv "$device_path" "$local_path"
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
    hdc -s "$HDC_ADDR" shell "rm -f $DEVICE_QWEN_DIR/*.json && rm -f $DEVICE_OUTPUT_DIR/*.txt"
fi

echo ""
echo "========================================"
echo "Baseline 处理完成!"
echo "输出文件已回收至: $LOCAL_OUTPUT_DIR"
echo "========================================"