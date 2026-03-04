import os
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer

from onnxruntime.quantization import matmul_nbits_quantizer, quant_utils
import onnxruntime as ort
import onnx


# =============== 配置区 ===============
model_id = "/data8/zhangdeming/models/microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

base_onnx_dir = Path("./onnx/onnx_llmlingua-2-xlm-roberta-large-meetingbank_fp32")
output_dir = Path("./onnx/onnx_llmlingua-2-xlm-roberta-large-meetingbank_int4")

onnx_filename = "model.onnx"
final_filename = "model_quantized.onnx"

# 量化参数
algorithm = "DEFAULT"  # ["DEFAULT", "RTN", "HQQ"]
bits = 4
op_types = ["MatMul"]  # 对于 transformer 模型，MatMul 是主要的量化目标
quant_axes = [0]  # 0: 按输出通道量化（weight矩阵的行）
block_size = 32  # 对于 4-bit 量化，建议使用较小的 block size
accuracy_level = 4  # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
quant_symmetric = False  # 对于 weight-only 量化，非对称通常效果更好
nodes_to_exclude = None  # 可以排除某些层不量化

# 额外配置
use_external_data = True  # 当模型 >2GB 时需要
skip_export_if_exists = True  # 如果 FP32 ONNX 已存在则跳过导出


def export_fp32_onnx():
    """导出 FP32 ONNX 模型"""
    base_onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = base_onnx_dir / onnx_filename

    if skip_export_if_exists and onnx_path.exists():
        print(f">>> 检测到已存在 ONNX: {onnx_path}，跳过导出。")
        # 验证 ONNX 文件是否有效
        try:
            onnx.checker.check_model(str(onnx_path))
            print(">>> ONNX 模型验证通过")
            return
        except Exception as e:
            print(f">>> 现有 ONNX 模型验证失败: {e}，将重新导出")

    print(">>> [1/2] 导出 FP32 ONNX ...")
    
    # 导出模型
    model = ORTModelForTokenClassification.from_pretrained(
        model_id, 
        export=True,
        force_download=False  # 避免重复下载
    )
    model.save_pretrained(base_onnx_dir)

    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(base_onnx_dir)

    # 验证导出的模型
    if not onnx_path.exists():
        raise FileNotFoundError(f"导出后未找到 {onnx_path}，请检查 Optimum 导出结果。")
    
    # 验证 ONNX 格式
    try:
        onnx.checker.check_model(str(onnx_path))
        print(">>> ONNX 模型导出成功且验证通过")
    except Exception as e:
        print(f">>> 警告：导出的 ONNX 模型验证失败: {e}")
        print(">>> 但量化过程可能仍可继续")


def quantize_int4_matmul():
    """执行 INT4 weight-only 量化"""
    print(">>> [2/2] 开始 ORT MatMul INT4 weight-only 量化 ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 复制除 model.onnx 外的其他文件
    for fn in base_onnx_dir.iterdir():
        if fn.name != onnx_filename:
            dst = output_dir / fn.name
            if fn.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(fn, dst)
            else:
                shutil.copy2(fn, dst)

    model_path = base_onnx_dir / onnx_filename
    quanted_model_path = output_dir / final_filename

    print(f">>> 加载模型: {model_path}")
    
    # 1) 读取 ONNX + shape inference（很关键）
    try:
        model = quant_utils.load_model_with_shape_infer(str(model_path))
        print(">>> 模型加载和 shape inference 成功")
    except Exception as e:
        print(f">>> 加载模型失败: {e}")
        print(">>> 尝试直接加载模型...")
        model = onnx.load(str(model_path))

    # 2) 创建 quant_axes 映射
    # 确保 op_types 和 quant_axes 长度匹配
    quant_axes_mapping = []
    for i, op_type in enumerate(op_types):
        axis = quant_axes[i] if i < len(quant_axes) else quant_axes[-1]
        quant_axes_mapping.append((op_type, axis))

    # 3) 选择算法配置
    if algorithm == "RTN":
        quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
        )
    elif algorithm == "HQQ":
        quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
            bits=bits,
            block_size=block_size,
            axis=quant_axes[0],  # HQQ 通常使用单个 axis
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
        )
    else:  # DEFAULT
        quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=quant_symmetric,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
            quant_axes=quant_axes_mapping,
        )

    # 设置 bits
    quant_config.bits = bits

    # 4) 运行量化
    print(f">>> 使用 {algorithm} 算法进行量化，bits={bits}，block_size={block_size}")
    
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model=model,
        block_size=block_size,
        is_symmetric=quant_symmetric,
        accuracy_level=accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=tuple(op_types),
        quant_axes=quant_axes_mapping,
        algo_config=quant_config,
        nodes_to_exclude=nodes_to_exclude,
    )

    print(">>> 开始量化处理...")
    quant.process()

    # 5) 保存
    print(f">>> 保存量化模型到: {quanted_model_path}")
    
    try:
        quant.model.save_model_to_file(
            str(quanted_model_path),
            use_external_data,  # save_as_external_data
        )
        print(f">>> 完成！量化模型已保存")
    except Exception as e:
        print(f">>> 保存模型失败: {e}")
        # 尝试不保存为 external data
        print(">>> 尝试不使用 external data 保存...")
        quant.model.save_model_to_file(
            str(quanted_model_path),
            False,  # save_as_external_data
        )
        print(f">>> 完成！量化模型已保存（不使用 external data）")
    
    # 验证量化后的模型
    try:
        onnx.checker.check_model(str(quanted_model_path))
        print(">>> 量化后的 ONNX 模型验证通过")
    except Exception as e:
        print(f">>> 警告：量化后的 ONNX 模型验证失败: {e}")


def test_quantized_model():
    """简单测试量化后的模型"""
    print("\n>>> [可选] 测试量化模型...")
    
    try:
        # 加载量化后的模型
        quanted_model_path = output_dir / final_filename
        
        # 创建推理会话
        session = ort.InferenceSession(str(quanted_model_path))
        
        # 获取输入输出信息
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f">>> 模型输入: {[i.name for i in inputs]}")
        print(f">>> 模型输出: {[o.name for o in outputs]}")
        
        # 创建一个简单的测试输入
        import numpy as np
        test_inputs = {}
        for input_info in inputs:
            # 创建 dummy 输入
            shape = [dim if isinstance(dim, int) else 1 for dim in input_info.shape]
            test_inputs[input_info.name] = np.random.randint(0, 1000, size=shape).astype(np.int64)
        
        # 运行推理
        results = session.run(None, test_inputs)
        print(f">>> 测试推理成功，输出形状: {[r.shape for r in results]}")
        
    except Exception as e:
        print(f">>> 测试失败: {e}")


if __name__ == "__main__":
    print("onnxruntime version:", ort.__version__)
    print(f"PyTorch 模型: {model_id}")
    print(f"FP32 ONNX 目录: {base_onnx_dir}")
    print(f"INT4 ONNX 目录: {output_dir}")
    
    # 导出 FP32 ONNX
    export_fp32_onnx()
    
    # 进行 INT4 量化
    quantize_int4_matmul()
    
    # 可选：测试量化后的模型
    # test_quantized_model()
    
    print("\n>>> 所有步骤完成！")
    print(f"FP32 模型: {base_onnx_dir / onnx_filename}")
    print(f"INT4 量化模型: {output_dir / final_filename}")