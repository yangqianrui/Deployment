import math
import torch
import torch.nn.functional as F
# 导入 PyTorch 的量化 functional API
import torch.nn.quantized.functional as F_q
import time 

try:
    from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
    _NUNCHAKU_AVAILABLE = True
except ImportError:
    svdq_gemm_w4a4_cuda = None
    _NUNCHAKU_AVAILABLE = False
    print("警告: Nunchaku CUDA kernel 未找到。W2/3/4A2/3/4 测试将被跳过。")
    
from quant.quantized_modules.robuq import (
    INT4_GROUP_SIZE, 
    _symmetric_int4_quant,
    _symmetric_int3_quant, 
    _symmetric_int2_quant  
)

GROUP_SIZE = INT4_GROUP_SIZE
CHANNEL_PAD = 128
BATCH_PAD = 256
N_RUNS = 100 # 测速运行次数
WARMUP_RUNS = 10 # 预热次数

# --- Nunchaku (W4A4 Kernel) 辅助函数 ---

def _unpack_weight(packed: torch.Tensor, N_pad: int, K: int) -> torch.Tensor:
    u8 = packed.view(N_pad, K // 2).to(torch.uint8)
    lo = (u8 & 0x0F).to(torch.int16)
    hi = (u8 >> 4).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    vals = torch.stack((lo, hi), dim=-1).reshape(N_pad, K)
    return vals.to(torch.float32)


def _unpack_activation(packed: torch.Tensor, M_pad: int, K: int) -> torch.Tensor:
    u8 = packed.view(M_pad, K // 2).to(torch.uint8)
    lo = (u8 & 0x0F).to(torch.int16)
    hi = (u8 >> 4).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    vals = torch.stack((lo, hi), dim=-1).reshape(M_pad, K)
    return vals.to(torch.float32)


def _broadcast_scales_w(wscales: torch.Tensor, N_pad: int, K: int) -> torch.Tensor:
    expanded = wscales.permute(1, 0).unsqueeze(-1).repeat(1, 1, GROUP_SIZE)
    return expanded.reshape(N_pad, K).to(torch.float32)


def _broadcast_scales_a(ascales: torch.Tensor, M_pad: int, K: int) -> torch.Tensor:
    expanded = ascales.permute(1, 0).unsqueeze(-1).repeat(1, 1, GROUP_SIZE)
    return expanded.reshape(M_pad, K).to(torch.float32)

# --- 基准测试函数 (Nunchaku W4A4 Kernel) ---

def run_benchmark_nunchaku(
    quant_fn: callable, 
    bit_name: str, 
    device: torch.device, 
    batch: int, 
    in_features: int, 
    out_features: int
) -> None:
    
    if not _NUNCHAKU_AVAILABLE:
        print(f"--- 跳过 {bit_name} (Nunchaku Kernel) 测试 ---")
        return

    print(f"\n--- 正在运行 {bit_name} 基准测试 (Nunchaku Kernel) ---")
    
    batch_pad = math.ceil(batch / BATCH_PAD) * BATCH_PAD
    k_pad = math.ceil(in_features / CHANNEL_PAD) * CHANNEL_PAD
    n_pad = math.ceil(out_features / CHANNEL_PAD) * CHANNEL_PAD

    activations = torch.randn(batch, in_features, device=device, dtype=torch.float16)
    act_pad = torch.zeros(batch_pad, k_pad, dtype=torch.float16, device=device)
    act_pad[:batch, :in_features] = activations

    weight = torch.randn(out_features, in_features, device=device, dtype=torch.float16)
    weight_pad = torch.zeros(n_pad, k_pad, dtype=torch.float16, device=device)
    weight_pad[:out_features, :in_features] = weight

    qweight, wscales = quant_fn(weight_pad.view(n_pad, k_pad // GROUP_SIZE, GROUP_SIZE))
    qact, ascales = quant_fn(act_pad.view(batch_pad, k_pad // GROUP_SIZE, GROUP_SIZE))

    qweight = qweight.to(torch.int8)
    qact = qact.to(torch.int8)

    # --- 精度验证 (Ref Sim) ---
    int_w = _unpack_weight(qweight, n_pad, k_pad)
    weight_sim = (_broadcast_scales_w(wscales, n_pad, k_pad) * int_w)[:out_features, :in_features]

    int_a = _unpack_activation(qact, batch_pad, k_pad)
    act_sim = (_broadcast_scales_a(ascales, batch_pad, k_pad) * int_a)[:batch, :in_features]

    ref_sim = act_sim @ weight_sim.t()

    out_pad = torch.zeros(batch_pad, n_pad, dtype=torch.float16, device=device)
    
    # --- 测速 ---
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 预热
    for _ in range(WARMUP_RUNS):
        svdq_gemm_w4a4_cuda(
            act=qact, wgt=qweight, out=out_pad, ascales=ascales, wscales=wscales,
            bias=None, lora_act_in=torch.empty(batch_pad, 0, dtype=torch.float32, device=device),
            lora_up=torch.empty(n_pad, 0, dtype=torch.float16, device=device),
            lora_down=None, lora_act_out=None, smooth_factor=None,
            act_unsigned=False, lora_scales=[], fp4=False,
        )
    torch.cuda.synchronize() 

    # 正式测速
    start.record()
    for _ in range(N_RUNS):
        svdq_gemm_w4a4_cuda(
            act=qact, wgt=qweight, out=out_pad, ascales=ascales, wscales=wscales,
            bias=None, lora_act_in=torch.empty(batch_pad, 0, dtype=torch.float32, device=device),
            lora_up=torch.empty(n_pad, 0, dtype=torch.float16, device=device),
            lora_down=None, lora_act_out=None, smooth_factor=None,
            act_unsigned=False, lora_scales=[], fp4=False,
        )
    end.record()
    torch.cuda.synchronize() 

    elapsed_ms = start.elapsed_time(end) / N_RUNS
    
    # --- 结果验证 ---
    out = out_pad[:batch, :out_features]
    diff = (out - ref_sim).float()
    max_abs_err = diff.abs().max().item()
    mean_abs_err = diff.abs().mean().item()
    rel_err = max_abs_err / ref_sim.abs().max().clamp_min(1e-6).item()

    print(f"[{bit_name}] 精度:")
    print(f"  max_abs_err: {max_abs_err:.6e}")
    print(f"  mean_abs_err: {mean_abs_err:.6e}")
    print(f"  relative_err: {rel_err:.6e}")
    print(f"[{bit_name}] 速度 (Avg over {N_RUNS} runs):")
    print(f"  Avg. Kernel Time: {elapsed_ms:.6f} ms")


# --- 通用辅助函数 (基于 PyTorch INT8 Kernel 的 N-bit 量化) ---

def _quantize_tensor_to_nbits(
    tensor: torch.Tensor, 
    bits: int, 
    eps: float = 1e-6
) -> torch.Tensor:
    if bits < 2 or bits > 8:
        raise ValueError(f"仅支持 2-8 bit 量化, 当前 bits={bits}")
    qmax = (1 << (bits - 1)) - 1
    if qmax <= 0:
        raise ValueError(f"bits={bits} 导致无效的量化级数")
    max_abs = tensor.abs().max()
    scale = float(max_abs) / qmax if max_abs > eps else 1.0
    return torch.quantize_per_tensor(tensor, scale, 0, torch.qint8)


def run_benchmark_torch_intx(
    bits: int,
    device: torch.device, 
    batch: int, 
    in_features: int, 
    out_features: int,
    note: str
) -> None:

    bit_name = f"W{bits}A{bits} ({note})"
    print(f"\n--- 正在运行 {bit_name} 基准测试 ---")

    # PyTorch 的 QInt 算子通常需要 float32 输入
    activations = torch.randn(batch, in_features, device=device, dtype=torch.float32)
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.float32)
    bias = torch.randn(out_features, device=device, dtype=torch.float32)

    # --- 精度验证 (Ref Sim) ---
    # 使用受限范围的 per-tensor 量化 (数值仍保存在 int8 中)
    q_act = _quantize_tensor_to_nbits(activations, bits)
    q_weight = _quantize_tensor_to_nbits(weight, bits)

    # 浮点参考 (使用反量化后的值)
    ref_sim = F.linear(q_act.dequantize(), q_weight.dequantize(), bias)

    # --- 准备 W8A8 运算 ---
    # --- 测速 ---
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 预热
    for _ in range(WARMUP_RUNS):
        # *** (修正) ***: 调用 torch.nn.quantized.functional.linear
        _ = F_q.linear(q_act, q_weight, bias)
    torch.cuda.synchronize()

    # 正式测速
    start.record()
    for _ in range(N_RUNS):
        # *** (修正) ***: 调用 torch.nn.quantized.functional.linear
        out_q = F_q.linear(q_act, q_weight, bias)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / N_RUNS
    
    # --- 结果验证 ---
    # 结果已经是量化后的，需要反量化
    out = out_q.dequantize() 

    diff = (out - ref_sim).float()
    max_abs_err = diff.abs().max().item()
    mean_abs_err = diff.abs().mean().item()
    rel_err = max_abs_err / ref_sim.abs().max().clamp_min(1e-6).item()

    print(f"[{bit_name}] 精度:")
    print(f"  max_abs_err: {max_abs_err:.6e}")
    print(f"  mean_abs_err: {mean_abs_err:.6e}")
    print(f"  relative_err: {rel_err:.6e}")
    print(f"[{bit_name}] 速度 (Avg over {N_RUNS} runs):")
    print(f"  Avg. Kernel Time: {elapsed_ms:.6f} ms")


def main() -> None:
    if not torch.cuda.is_available():
        print("错误：需要 CUDA 设备来运行此基准测试")
        return

    device = torch.device("cuda")
    torch.manual_seed(0)

    # 增加维度以便观察测速
    batch = 1024
    in_features = 4096
    out_features = 4096

    print(f"设备: {device}")
    print(f"配置: Batch={batch}, In={in_features}, Out={out_features}")
    print(f"Nunchaku Pad: BATCH_PAD={BATCH_PAD}, CHANNEL_PAD={CHANNEL_PAD}, GROUP_SIZE={GROUP_SIZE}")
    print(f"测速: Warmup={WARMUP_RUNS} runs, Benchmark={N_RUNS} runs")

    # 运行 W2A2 (Nunchaku)
    run_benchmark_nunchaku(
        quant_fn=_symmetric_int2_quant, 
        bit_name="W2A2 (Simulated on W4A4 Kernel)", 
        device=device, batch=batch, in_features=in_features, out_features=out_features
    )

    # 运行 W3A3 (Nunchaku)
    run_benchmark_nunchaku(
        quant_fn=_symmetric_int3_quant, 
        bit_name="W3A3 (Simulated on W4A4 Kernel)", 
        device=device, batch=batch, in_features=in_features, out_features=out_features
    )

    # 运行 W4A4 (Nunchaku)
    run_benchmark_nunchaku(
        quant_fn=_symmetric_int4_quant, 
        bit_name="W4A4 (Nunchaku Native)", 
        device=device, batch=batch, in_features=in_features, out_features=out_features
    )
    
    # 运行 W5A5 (基于 INT8 Kernel)
    run_benchmark_torch_intx(
        bits=5,
        device=device, batch=batch, in_features=in_features, out_features=out_features,
        note="Simulated on Torch INT8 Kernel"
    )

    # 运行 W6A6 (基于 INT8 Kernel)
    run_benchmark_torch_intx(
        bits=6,
        device=device, batch=batch, in_features=in_features, out_features=out_features,
        note="Simulated on Torch INT8 Kernel"
    )

    # 运行 W8A8 (Torch Native)
    run_benchmark_torch_intx(
        bits=8,
        device=device, batch=batch, in_features=in_features, out_features=out_features,
        note="Torch Native"
    )

if __name__ == "__main__":
    main()
