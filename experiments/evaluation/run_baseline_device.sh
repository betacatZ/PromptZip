source ~/.zshrc
# 兼容 zsh 和 bash

# ================================================
# Baseline端侧推理自动化脚本（Qwen2.5-7b-instruct）
# 功能：
# 1. 发送 baseline JSON 文件到设备
# 2. 在设备上逐条运行 qwen 推理
# 3. 回收输出文件
#
# 支持一次输入多个 LOCAL_INPUT_DIR，输出保持对应目录结构
# 使用方式：
#   ./run_baseline_device.sh -i dir1 dir2 dir3
#   ./run_baseline_device.sh -i dir1 -i dir2 -i dir3
# ================================================

set -e

# 设置 shell 兼容性选项（对于 zsh）
if [ -n "$ZSH_VERSION" ]; then
    setopt NO_NOMATCH  # 在无匹配文件时不报错
fi

# 配置（请根据实际情况修改）
DEFAULT_INPUT_DIRS="/data8/zhangdeming/PromptZip/GEWU-dataset-baseline/test"  # 默认本地输入目录
DEVICE_QWEN_DIR="/data/qwen2"                                                  # 设备上 qwen 工作目录
DEVICE_TEST_DIR="/data/qwen2/test"                                              # 设备上的测试JSON目录
DEVICE_OUTPUT_DIR="/data/qwen2/output/test"                                     # 设备上的输出目录
DEFAULT_OUTPUT_DIR="/data8/zhangdeming/PromptZip/GEWU_output/baseline"         # 默认本地回收文件的根目录
HDC_ADDR="100.103.109.221:8710"                                                 # hdc 连接地址

# 输入目录数组
INPUT_DIRS=()
LOCAL_OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -i, --input DIR...  本地包含 baseline JSON 文件的目录（可指定多个）"
    echo "                      多个目录可连续传入: -i dir1 dir2 dir3"
    echo "                      或多次传入: -i dir1 -i dir2"
    echo "                      输出会保持对应的目录结构（basename作为子目录）"
    echo "  -o, --output DIR    本地回收文件的根目录（默认: $DEFAULT_OUTPUT_DIR）"
    echo "  -h, --help          显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -i /path/to/test1 /path/to/test2"
    echo "      回收到: $DEFAULT_OUTPUT_DIR/test1/ 和 $DEFAULT_OUTPUT_DIR/test2/"
}

# 解析命令行参数（支持 -i 后跟多个目录直到下一个选项）
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input)
            shift
            while [[ "$#" -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                INPUT_DIRS+=("$1")
                shift
            done
            ;;
        -o|--output) LOCAL_OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "未知参数: $1"; show_help; exit 1 ;;
    esac
done

# 若未指定输入目录，使用默认值
if [ ${#INPUT_DIRS[@]} -eq 0 ]; then
    INPUT_DIRS=("$DEFAULT_INPUT_DIRS")
fi

echo "========================================"
echo "Baseline 端侧推理自动化脚本"
echo "========================================"
echo "本地输入目录:"
for dir in "${INPUT_DIRS[@]}"; do
    subdir=$(basename "$dir")
    echo "  $dir -> $LOCAL_OUTPUT_DIR/$subdir/"
done
echo "设备上 qwen 目录: $DEVICE_QWEN_DIR"
echo ""

# 1. 检查所有本地输入目录
for dir in "${INPUT_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "错误: 本地输入目录不存在: $dir"
        exit 1
    fi
done

# 2. 准备本地输出目录（每个输入目录对应一个子目录）
for dir in "${INPUT_DIRS[@]}"; do
    subdir=$(basename "$dir")
    mkdir -p "$LOCAL_OUTPUT_DIR/$subdir"
done

# 3. 在设备上创建目录
echo "在设备上创建目录..."
hdc -s "$HDC_ADDR" shell "mkdir -p $DEVICE_TEST_DIR && mkdir -p $DEVICE_OUTPUT_DIR"

# 4. 发送所有 baseline JSON 文件到设备的测试目录
echo "发送 JSON 文件到设备..."
total_files=0
for dir in "${INPUT_DIRS[@]}"; do
    count=$(ls "$dir"/*.json 2>/dev/null | wc -l)
    total_files=$((total_files + count))
done

if [ "$total_files" -eq 0 ]; then
    echo "  警告: 所有输入目录中没有找到 JSON 文件"
    exit 0
fi

echo "  共 $total_files 个 JSON 文件"
current=0
for dir in "${INPUT_DIRS[@]}"; do
    subdir=$(basename "$dir")
    for json_file in "$dir"/*.json; do
        if [ -e "$json_file" ]; then
            current=$((current + 1))
            filename=$(basename "$json_file")
            echo "  [$current/$total_files] [$subdir] 发送: $filename"
            hdc -s "$HDC_ADDR" file send "$json_file" "$DEVICE_TEST_DIR/$filename"
        fi
    done
done

# 5. 在设备上逐条运行推理
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
printf "%s\n" "# 统计总数" >> "$TEMP_SCRIPT"
printf "%s\n" "total=0" >> "$TEMP_SCRIPT"
printf "%s\n" "for f in $DEVICE_TEST_DIR/*.json; do" >> "$TEMP_SCRIPT"
printf "%s\n" "    if [ -f \"\$f\" ]; then total=\$((total + 1)); fi" >> "$TEMP_SCRIPT"
printf "%s\n" "done" >> "$TEMP_SCRIPT"
printf "%s\n" "current=0" >> "$TEMP_SCRIPT"
printf "%s\n" "for json_file in $DEVICE_TEST_DIR/*.json; do" >> "$TEMP_SCRIPT"
printf "%s\n" "    if [ -f \"\$json_file\" ]; then" >> "$TEMP_SCRIPT"
printf "%s\n" "        current=\$((current + 1))" >> "$TEMP_SCRIPT"
printf "%s\n" "        filename=\$(basename \"\$json_file\")" >> "$TEMP_SCRIPT"
printf "%s\n" "        stem=\$(basename \"\$json_file\" .json)" >> "$TEMP_SCRIPT"
printf "%s\n" "        output_file=\"$DEVICE_OUTPUT_DIR/\$stem.txt\"" >> "$TEMP_SCRIPT"
printf "%s\n" "        " >> "$TEMP_SCRIPT"
printf "%s\n" "        echo \"[\$current/\$total] 正在处理: \$filename\"" >> "$TEMP_SCRIPT"
printf "%s\n" "        " >> "$TEMP_SCRIPT"
printf "%s\n" "        # 执行推理" >> "$TEMP_SCRIPT"
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

# 6. 回收输出文件（按输入目录的basename分子目录回收）
echo ""
echo "回收输出文件..."
current=0
for dir in "${INPUT_DIRS[@]}"; do
    subdir=$(basename "$dir")
    for json_file in "$dir"/*.json; do
        if [ -e "$json_file" ]; then
            current=$((current + 1))
            filename=$(basename "$json_file" .json)
            output_file="$filename.txt"
            device_path="$DEVICE_OUTPUT_DIR/$output_file"
            local_path="$LOCAL_OUTPUT_DIR/$subdir/$output_file"

            echo "  [$current/$total_files] [$subdir] 回收: $output_file"
            hdc -s "$HDC_ADDR" file recv "$device_path" "$local_path"
        fi
    done
done

# 7. 清理设备上的测试文件（可选）
read -p "是否清理设备上的测试文件？(y/N): " clean_confirm
if [[ "$clean_confirm" =~ ^[Yy]$ ]]; then
    echo "清理设备上的测试文件..."
    hdc -s "$HDC_ADDR" shell "rm -f $DEVICE_TEST_DIR/*.json && rm -f $DEVICE_OUTPUT_DIR/*.txt"
fi

echo ""
echo "========================================"
echo "Baseline 处理完成!"
echo "输出文件已回收至:"
for dir in "${INPUT_DIRS[@]}"; do
    subdir=$(basename "$dir")
    echo "  $LOCAL_OUTPUT_DIR/$subdir/"
done
echo "========================================"