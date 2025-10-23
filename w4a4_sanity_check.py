import math

import torch

from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
from quant.quantized_modules.robuq import INT4_GROUP_SIZE, _symmetric_int4_quant

GROUP_SIZE = INT4_GROUP_SIZE
CHANNEL_PAD = 128
BATCH_PAD = 256


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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备来运行 W4A4 对拍测试")

    device = torch.device("cuda")
    torch.manual_seed(0)

    batch = 192
    in_features = 96
    out_features = 64

    batch_pad = math.ceil(batch / BATCH_PAD) * BATCH_PAD
    k_pad = math.ceil(in_features / CHANNEL_PAD) * CHANNEL_PAD
    n_pad = math.ceil(out_features / CHANNEL_PAD) * CHANNEL_PAD

    activations = torch.randn(batch, in_features, device=device, dtype=torch.float16)
    act_pad = torch.zeros(batch_pad, k_pad, dtype=torch.float16, device=device)
    act_pad[:batch, :in_features] = activations

    weight = torch.randn(out_features, in_features, device=device, dtype=torch.float16)
    weight_pad = torch.zeros(n_pad, k_pad, dtype=torch.float16, device=device)
    weight_pad[:out_features, :in_features] = weight

    qweight, wscales = _symmetric_int4_quant(weight_pad.view(n_pad, k_pad // GROUP_SIZE, GROUP_SIZE))

    qact, ascales = _symmetric_int4_quant(act_pad.view(batch_pad, k_pad // GROUP_SIZE, GROUP_SIZE))

    qweight = qweight.to(torch.int8)
    qact = qact.to(torch.int8)

    int_w = _unpack_weight(qweight, n_pad, k_pad)
    weight_sim = (_broadcast_scales_w(wscales, n_pad, k_pad) * int_w)[:out_features, :in_features]

    int_a = _unpack_activation(qact, batch_pad, k_pad)
    act_sim = (_broadcast_scales_a(ascales, batch_pad, k_pad) * int_a)[:batch, :in_features]

    ref_sim = act_sim @ weight_sim.t()

    out_pad = torch.zeros(batch_pad, n_pad, dtype=torch.float16, device=device)
    svdq_gemm_w4a4_cuda(
        act=qact,
        wgt=qweight,
        out=out_pad,
        ascales=ascales,
        wscales=wscales,
        bias=None,
        lora_act_in=torch.empty(batch_pad, 0, dtype=torch.float32, device=device),
        lora_up=torch.empty(n_pad, 0, dtype=torch.float16, device=device),
        lora_down=None,
        lora_act_out=None,
        smooth_factor=None,
        act_unsigned=False,
        lora_scales=[],
        fp4=False,
    )

    out = out_pad[:batch, :out_features]

    diff = (out - ref_sim).float()
    max_abs_err = diff.abs().max().item()
    mean_abs_err = diff.abs().mean().item()
    rel_err = max_abs_err / ref_sim.abs().max().clamp_min(1e-6).item()

    print(f"max_abs_err: {max_abs_err:.6e}")
    print(f"mean_abs_err: {mean_abs_err:.6e}")
    print(f"relative_err: {rel_err:.6e}")


if __name__ == "__main__":
    main()
