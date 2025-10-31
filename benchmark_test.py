import torch
import torch.nn as nn
import time
import math

# 假设您的量化模块在 quant 目录下
from quant.quantized_modules.robuq import RobuQLinear, QuantAct, UniformQuantSTE

# --- 配置 ---
BATCH_SIZE = 1024
IN_FEATURES = 4096
OUT_FEATURES = 4096
N_RUNS = 100 # 测速运行次数
WARMUP_RUNS = 10 # 预热次数
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def benchmark_fp(bits: int, device: torch.device, batch: int, in_features: int, out_features: int) -> float:
    """
    测试 RobuQLinear._forward_fp 方法的速度。
    """
    print(f"\n--- 正在测试 W{bits}A{bits} (FP Forward - 原始方法) ---")
    
    # 1. 实例化模型
    # 确保使用 UniformQuantSTE 进行量化，nbits 和 w_bits 相同
    # 您的 robuq.py 中的 UNIFORM_QUANT_TABLE 包含了 1-8 bit
    model = RobuQLinear(
        in_features, 
        out_features, 
        bias=False, 
        if_lora=False, 
        if_hadamard=False, 
        n_bits=bits, 
        w_bits=bits 
    ).to(device).half() # 使用半精度以模拟典型推理场景
    
    # 2. 创建输入数据
    input_tensor = torch.randn(batch, in_features, device=device, dtype=torch.float16)
    
    # 3. 设置模型为评估模式并禁用梯度
    model.eval()
    
    # 4. 预热
    print(f"进行 {WARMUP_RUNS} 次预热...")
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model._forward_fp(input_tensor) # 直接调用 _forward_fp
    torch.cuda.synchronize(device=device) 
    print("预热完成.")

    # 5. 正式测速
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"开始 {N_RUNS} 次测速...")
    start_event.record()
    with torch.no_grad():
        for _ in range(N_RUNS):
            _ = model._forward_fp(input_tensor) # 直接调用 _forward_fp
    end_event.record()
    torch.cuda.synchronize(device=device) 
    print("测速完成.")

    elapsed_ms = start_event.elapsed_time(end_event) / N_RUNS
    print(f"W{bits}A{bits} (原始方法) 平均执行时间: {elapsed_ms:.6f} ms")
    return elapsed_ms

def main():
    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA 设备，将在 CPU 上运行（速度较慢且不准确）！")
        global DEVICE
        DEVICE = torch.device("cpu") 

    print(f"设备: {DEVICE}")
    print(f"配置: Batch={BATCH_SIZE}, In={IN_FEATURES}, Out={OUT_FEATURES}")
    print(f"测速: Warmup={WARMUP_RUNS} runs, Benchmark={N_RUNS} runs")

    results = {}
    # 新增 8-bit 到测试列表
    for bits in [2, 3, 4, 5, 6, 8]:
        avg_time = benchmark_fp(bits, DEVICE, BATCH_SIZE, IN_FEATURES, OUT_FEATURES)
        results[f"W{bits}A{bits} (FP)"] = avg_time
        
    print("\n--- 测速结果总结 (原始方法) ---")
    for config, avg_time in results.items():
        print(f"{config}: {avg_time:.6f} ms")

if __name__ == "__main__":
    main()
