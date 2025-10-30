import math
import warnings
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .hadamard_utils import *

if TYPE_CHECKING:
    from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda as _svdq_gemm_w4a4_cuda
else:  # pragma: no cover - runtime import
    _svdq_gemm_w4a4_cuda = None

try:  # pragma: no cover - optional dependency
    from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda

    _NUNCHAKU_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    svdq_gemm_w4a4_cuda = None
    _NUNCHAKU_AVAILABLE = False

INT4_GROUP_SIZE = 64
# 定义不同位宽的存储最大值 (2^(N-1) - 1)
INT2_STORAGE_MAX = (1 << 1) - 1  # 1
INT3_STORAGE_MAX = (1 << 2) - 1  # 3
INT4_STORAGE_MAX = (1 << 3) - 1  # 7, 原始值
W4A4_CHANNEL_PAD = 128
W4A4_BATCH_PAD = 256
# 预计算的均匀量化表
#-----------Gaussian Optimal Uniform Quantization--------------
UNIFORM_QUANT_TABLE = {
    1: {'a_opt': 0.797885, 'delta': 0.797885, 'mse': 0.36338023},
    2: {'a_opt': 1.991374, 'delta': 0.995687, 'mse': 0.11884605},
    3: {'a_opt': 2.344078, 'delta': 0.586019, 'mse': 0.03743966},
    4: {'a_opt': 2.681605, 'delta': 0.335201, 'mse': 0.01154288},
    5: {'a_opt': 3.010220, 'delta': 0.188139, 'mse': 0.00349521},
    6: {'a_opt': 3.330017, 'delta': 0.104063, 'mse': 0.00104005},
    7: {'a_opt': 3.639531, 'delta': 0.056868, 'mse': 0.00030433},
    8: {'a_opt': 3.937585, 'delta': 0.030762, 'mse': 0.00008769}
}



def _pack_int4_blocks(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values into uint8 (for nunchaku compatibility).

    Args:
        q: Tensor with shape (B, G, 64) and dtype int16 containing values in [-8, 7].

    Returns:
        Packed tensor with shape (B, G * 32) and dtype uint8.
    """

    if q.dtype not in (torch.int16, torch.int32, torch.int64):
        raise TypeError("Expected signed integer tensor for int4 packing.")
    if q.shape[-1] != 64:
        raise ValueError("Last dimension must be 64 for int4 packing.")

    q_view = q.view(*q.shape[:-1], 32, 2)
    low = (q_view[..., 0] & 0xF).to(torch.uint8)
    high = (q_view[..., 1] & 0xF).to(torch.uint8)
    packed = low | (high << 4)
    return packed.reshape(q.shape[0], -1)


def _symmetric_int4_quant(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize grouped values to int4 with per-sample scales.

    Args:
        groups: Tensor with shape (B, G, 64) and dtype float16/float32.

    Returns:
        packed_q: int8 tensor with shape (B, G * 32).
        scales: float16 tensor with shape (G, B).
    """

    if groups.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError("Expected floating tensor for int4 quantization.")

    max_abs = groups.abs().amax(dim=-1)
    scale = max_abs / float(INT4_STORAGE_MAX)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    q = torch.round(groups / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int16)
    packed = _pack_int4_blocks(q)
    return packed.contiguous(), scale.transpose(0, 1).contiguous().to(torch.float16)

def _symmetric_int3_quant(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize grouped values to int3, but pack as int4 for compatibility.
    Upscales quantized values by 2 (range [-8, 6]) and adjusts scale by / 2.0.
    """
    if groups.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError("Expected floating tensor for int3 quantization.")

    max_abs = groups.abs().amax(dim=-1)
    scale = max_abs / float(INT3_STORAGE_MAX) # scale by 3
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    
    # Quantize to [-4, 3]
    q = torch.round(groups / scale.unsqueeze(-1)).clamp(-4, 3)
    
    # Upscale to 4-bit range [-8, 6]
    q_upscaled = (q * 2).to(torch.int16)
    
    # Adjust scale
    scale_adjusted = scale / 2.0
    
    packed = _pack_int4_blocks(q_upscaled)
    return packed.contiguous(), scale_adjusted.transpose(0, 1).contiguous().to(torch.float16)

def _symmetric_int2_quant(groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize grouped values to int2, but pack as int4 for compatibility.
    Upscales quantized values by 4 (range [-8, 4]) and adjusts scale by / 4.0.
    """
    if groups.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError("Expected floating tensor for int2 quantization.")

    max_abs = groups.abs().amax(dim=-1)
    # For 2 bits (4 levels), max abs value is 1 (if range is [-2, 1]) 
    # or 2 if we allow [-2, 1]. Let's use 1 ([-2, 1] -> 3 levels?)
    # Let's check int4: max_abs / 7. q.clamp(-8, 7).
    # Let's use 2 levels: [-1, 1]. Max abs = 1.
    # storage_max = 1 ([-2, 1] range has 4 values, but max abs is 2?)
    # Let's assume 2 bits = 4 levels = -2, -1, 0, 1. Max storage = 1.5?
    # Let's stick to the pattern: 2^(N-1) - 1
    # N=2 -> 2^1 - 1 = 1.
    scale = max_abs / float(INT2_STORAGE_MAX) # scale by 1
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    
    # Quantize to [-2, 1] (This is 4 levels, but asymmetric. Let's try [-1, 1] (3 levels))
    # Let's use [-2, 1] (4 levels)
    q = torch.round(groups / scale.unsqueeze(-1)).clamp(-2, 1) 
    
    # Upscale to 4-bit range [-8, 4]
    q_upscaled = (q * 4).to(torch.int16)
    
    # Adjust scale
    scale_adjusted = scale / 4.0

    packed = _pack_int4_blocks(q_upscaled)
    return packed.contiguous(), scale_adjusted.transpose(0, 1).contiguous().to(torch.float16)

class UniformQuantSTE(nn.Module):
    def __init__(self, bits):
        """
        精确的均匀量化器(带STE梯度估计)
        
        Args:
            bits: 量化位数 (1-8)
        """
        super().__init__()
        assert bits in UNIFORM_QUANT_TABLE, f"Unsupported bit-width {bits}, must be 1-8"
        
        self.bits = bits
        self.config = UNIFORM_QUANT_TABLE[bits]
        
        # 计算量化参数
        self.a_opt = self.config['a_opt']
        self.delta = self.config['delta']
        
        # 注册为buffer确保设备安全
        self.register_buffer('min_val', torch.tensor(-self.a_opt))
        self.register_buffer('max_val', torch.tensor(self.a_opt))
        self.register_buffer('delta_tensor', torch.tensor(self.delta))
        
        # 1-bit特殊处理
        if bits == 1:
            self.register_buffer('level_neg', torch.tensor(-self.a_opt))
            self.register_buffer('level_pos', torch.tensor(self.a_opt))
        
    def forward(self, x):
        # 设备同步
        self.min_val = self.min_val.to(x.device)
        self.max_val = self.max_val.to(x.device)
        self.delta_tensor = self.delta_tensor.to(x.device)
        
        # 1-bit特殊处理
        if self.bits == 1:
            return self._quantize_1bit(x)
        
        # 截断到最优范围
        x_clipped = torch.clamp(x, self.min_val, self.max_val)
        
        # 量化
        x_quant = torch.round(x_clipped / self.delta_tensor)
        x_quant = torch.clamp(x_quant, 
                              -2**(self.bits-1), 
                              2**(self.bits-1)-1)
        
        # 反量化 (STE)
        x_dequant = x_quant * self.delta_tensor
        return x + (x_dequant - x).detach()
    
    def _quantize_1bit(self, x):
        """1-bit量化特殊实现"""
        # 设备同步
        self.level_neg = self.level_neg.to(x.device)
        self.level_pos = self.level_pos.to(x.device)
        
        # 量化
        quantized = torch.where(x < 0, self.level_neg, self.level_pos)
        
        # STE
        return x + (quantized - x).detach()
    
    def extra_repr(self):
        return (f"bits={self.bits}, Δ={self.delta:.6f}, "
                f"range=[{-self.a_opt:.4f}, {self.a_opt:.4f}], "
                f"MSE={self.config['mse']:.6f}")

class QuantAct(nn.Module):
    def __init__(self, quant_func=None):
        super().__init__()
        # 默认量化函数为恒等操作
        self.quant_func = quant_func if quant_func is not None else self.identity
    
    def identity(self, x):
        """占位量化函数，直接返回输入"""
        return x

    def forward(self, x, dim=-1):
        """
        x: 输入张量
        dim: 指定量化操作的通道维度
        """
        # ==== 步骤1: 对称化处理 ====
        # 沿指定维度计算均值 (保持维度以支持广播)
        mean = x.mean(dim=dim, keepdim=True)
        x_sym = x - mean  # 中心化处理 (均值为0)
        # ==== 步骤2: 基于MAD估计方差 ====
        # 计算平均绝对偏差 (MAD)
        abs_mean = torch.abs(x_sym).mean(dim=dim, keepdim=True)
        # 通过MAD估计标准差 (σ = MAD * sqrt(π/2))
        std = abs_mean * math.sqrt(math.pi / 2) # 
        #直接计算std
        # ==== 步骤3: 数据归一化 ====
        x_norm = x_sym / std

        # ==== 步骤4: 应用量化函数 ====
        x_quant = self.quant_func(x_norm)
        
        # ==== 步骤5: 反归一化 ====
        x_denorm = x_quant * std
        
        # ==== 步骤6: 恢复原始分布 ====
        x_recon = x_denorm + mean
        
        return x_recon

#------------------------------Low rank branch-----------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, pretrained_weight=None):
        super(LoRALinear, self).__init__()

        # 确保 rank 在合理范围内
        self.rank = min(rank, in_features, out_features)

        # 定义低秩线性层 A 和 B
        self.B = nn.Linear(in_features, self.rank, bias=False)  # [B: in_features -> rank]
        self.A = nn.Linear(self.rank, out_features, bias=False)  # [A: rank -> out_features]

        if pretrained_weight is not None:
            # 检查预训练权重矩阵的形状
            if pretrained_weight.ndimension() == 2 and pretrained_weight.shape == (out_features, in_features):
                # 对预训练权重进行 SVD 分解
                U, S, V = torch.svd(pretrained_weight)
                # 使用前 rank 个奇异值来初始化 A 和 B
                with torch.no_grad():
                    self.A.weight.data = U[:, :self.rank] @ torch.diag(torch.sqrt(S[:self.rank]))  # [out_features, rank]
                    self.B.weight.data = (torch.diag(torch.sqrt(S[:self.rank])) @ V[:, :self.rank].t())  # [rank, in_features]
            else:
                raise ValueError("预训练权重形状不匹配。应为 (out_features, in_features) 矩阵。")

        else:
            # 如果没有预训练权重，使用默认初始化
            # nn.Linear 默认使用 Xavier 初始化
            nn.init.normal_(self.A.weight, mean=0.0, std=1e-4)

    def forward(self, x):
        # LoRA的低秩适配操作
        out = self.A(self.B(x))  # 先通过 B，再通过 A，低秩适配
        return out

    def get_equiv_weight(self):
        # 返回等效的权重矩阵 A @ B
        #print(self.A.weight.size(), self.B.weight.size())
        return torch.matmul(self.A.weight, self.B.weight)
#------------------------------Ternary Quantization------------------


class TernaryPerChannel(Function):
    """
    Per-Channel三值量化 (Ternary Quantization)
    对每个输出通道独立计算:
    1. 三值化阈值 (τ = 0.5 * E[|W_c|])
    2. 最优缩放因子 (α_c = E[|W_c| | |W_c| ≥ τ])
    反向传播时应用梯度裁剪（Gradient Clipping）
    """
    @staticmethod
    def forward(ctx, weight, epsilon=1e-6, clip_value=1.0):
        """
        前向传播:
        1. 计算通道级阈值 τ_c 和最优缩放因子 α_c
        2. 三值化权重: {-1, 0, +1} * α_c
        
        参数:
            weight: 待量化的权重张量 [out_features, in_features]
            epsilon: 防止除零的小常数
            clip_value: 梯度裁剪的阈值
        """
        ctx.save_for_backward(weight)
        ctx.epsilon = epsilon
        ctx.clip_value = clip_value
        
        # --- 计算阈值和缩放因子 ---
        abs_weight = torch.abs(weight)
        tau = 0.5 * torch.mean(abs_weight, dim=-1, keepdim=True)  # 阈值 τ_c = 0.5 * E[|W_c|]
        
        # 计算非零权重数量 N_c [C, 1]
        mask = (abs_weight >= tau).float()
        N_c = torch.sum(mask, dim=-1, keepdim=True)
        N_c = torch.clamp(N_c, min=1.0)  # 避免除零
        
        # 计算最优 α_c = E[|W_c| | |W_c| ≥ τ] [C, 1]
        alpha = torch.sum(abs_weight * mask, dim=-1, keepdim=True) / N_c
        
        # --- 三值化 ---
        ternary_weight = torch.where(
            abs_weight >= tau,
            torch.sign(weight),
            torch.zeros_like(weight)
        )
        scaled_weight = alpha * ternary_weight  # 最终量化结果: {-α_c, 0, +α_c}
        
        # 保存中间结果用于反向传播
        ctx.alpha = alpha
        ctx.mask = mask
        
        return scaled_weight

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播 (STE + Gradient Clipping):
        1. 直通估计器（STE）传递梯度
        2. 对梯度进行裁剪，抑制噪声
        """
        weight, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # 梯度裁剪（限制在 [-clip_value, clip_value]）
        grad_input = torch.clamp(grad_input, -ctx.clip_value, ctx.clip_value)
        
        return grad_input, None, None  # 对 epsilon 和 clip_value 返回 None

def quantize_linear_weights_perchannel(weight, clip_value=1.0):
    """
    对线性层权重进行Per-Channel三值量化（带梯度裁剪）
    
    参数:
        weight: 线性层的权重张量 [out_features, in_features]
        clip_value: 梯度裁剪的阈值（默认1.0）
    
    返回:
        quantized_weight: 量化后的权重 {-α_c, 0, +α_c}
    """
#    assert weight.dim() == 2, "权重必须是2D张量 [out_features, in_features]"
    return TernaryPerChannel.apply(weight, 1e-6, clip_value)
class RobuQLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        pretained_weight=None,
        if_lora=True,
        if_hadamard=False,
        n_bits=4,
        w_bits=1.58,
    ):
        super(RobuQLinear, self).__init__(in_features, out_features, bias=bias)
        if pretained_weight is not None:
            self.weight = pretained_weight
        self.n_bits = n_bits
        self.w_bits = w_bits
        self.if_hadamard = if_hadamard
        if if_hadamard:
            with torch.no_grad():
                self.weight.copy_(self.hadamard_init(in_features, self.weight.data))
        self.rank = 16
        self.if_lora = False
        if if_lora and self.rank < min(in_features, out_features):
            self.if_lora = True
        if self.if_lora:
            self.lora = LoRALinear(in_features, out_features, rank=self.rank, pretrained_weight=self.weight).to(
                self.weight.device
            )
            self.weight = nn.Parameter(self.weight.data - self.lora.get_equiv_weight())
        if w_bits == 1.58:
            self.weight_quantize = quantize_linear_weights_perchannel
        else:
            self.weight_quantize = QuantAct(quant_func=UniformQuantSTE(bits=w_bits))
        if n_bits == 32:
            self.activation_quantize = nn.Identity()
        else:
            self.activation_quantize = QuantAct(quant_func=UniformQuantSTE(bits=n_bits))
        self.cin = in_features
        self.cout = out_features
        # 仅当 A 和 W 都是4bit时，才启用Nunchaku W4A4
        self._use_w4a4 = _NUNCHAKU_AVAILABLE and n_bits == 4 and w_bits == 4
        # 针对 2/3 bit，我们复用 4-bit 算子，因此也设置 _use_w4a4
        if _NUNCHAKU_AVAILABLE and n_bits in [2, 3] and w_bits in [2, 3] and n_bits == w_bits:
             self._use_w4a4 = True
             
        self._w4a4_ready = False
        self._k_pad = math.ceil(self.cin / W4A4_CHANNEL_PAD) * W4A4_CHANNEL_PAD if self._use_w4a4 else self.cin
        self._n_pad = math.ceil(self.cout / W4A4_CHANNEL_PAD) * W4A4_CHANNEL_PAD if self._use_w4a4 else self.cout
        self._w4a4_qweight = None
        self._w4a4_wscales = None
        self._w4a4_bias = None
        
        # 根据位宽选择W4A4兼容的量化函数
        if self._use_w4a4:
            if n_bits == 4 and w_bits == 4:
                self._quant_fn_w4a4 = _symmetric_int4_quant
            elif n_bits == 3 and w_bits == 3:
                self._quant_fn_w4a4 = _symmetric_int3_quant
            elif n_bits == 2 and w_bits == 2:
                self._quant_fn_w4a4 = _symmetric_int2_quant
            else:
                # 混合精度 W4A2 W4A3 等暂不启用
                self._use_w4a4 = False

    def hadamard_init(self,in_features,weight):
        """
    对传入的权重矩阵执行随机哈达玛变换，并保存随机列向量供前向传播使用
    
    参数:
        in_features (int): 输入特征维度
        weight (torch.Tensor): 需要变换的权重矩阵
    
    返回:
        torch.Tensor: 经过哈达玛变换后的权重矩阵
        """
        hadK, K = get_hadK(in_features)  # n 是输入维度
        # 对传入的权重矩阵执行随机哈达玛变换，并保存下随机列向量
        random_signs = torch.randint(0, 2, (in_features,)).float() * 2 - 1
        # 保存随机列向量，用于前向传播中的逆变换
        self.register_buffer('hadamard_signs', random_signs.to(weight.device))
        # 使用哈达玛矩阵进行变换
        # 应用随机符号向量作为对角矩阵与权重相乘
        diagonal_matrix = torch.diag(random_signs)
        diagonal_matrix = diagonal_matrix.to(weight.device)  # 确保对角矩阵在正确的设备上
        # 应用哈达玛变换到权重矩阵
        # 这里我们使用 matmul_hadU 函数来执行哈达玛变换
        transformed_weight = matmul_hadU_cuda(( weight@diagonal_matrix).cuda(),hadK,K)
        self.hadamard_signs.to('cuda')  # 确保随机列向量在正确的设备上
        return transformed_weight
    def activation_hadamard(self, input):
        if self.hadamard_signs is not None:
            self.hadamard_signs = self.hadamard_signs.to(input.device)  # 确保符号向量在正确的设备上
            # 根据输入的维度直接在最后一个维度上应用符号变换
            if input.dim() == 2:  # [batch_size, features]
                input = input * self.hadamard_signs.unsqueeze(0)
            elif input.dim() == 3:  # [batch_size, seq_length, features]
                input = input * self.hadamard_signs.unsqueeze(0).unsqueeze(1)
            elif input.dim() == 4:  # [batch_size, other_dim1, other_dim2, features]
                input = input * self.hadamard_signs.unsqueeze(0).unsqueeze(1).unsqueeze(2)
            hadK, K = get_hadK(self.cin) 
            input = matmul_hadU_cuda(input.cuda(),hadK,K)
        return input
    def refresh_bits(self,a_bits,w_bits):
        self.activation_quantize = QuantAct(quant_func=UniformQuantSTE(bits=a_bits))
        self.weight_quantize = QuantAct(quant_func=UniformQuantSTE(bits=w_bits))
        self.n_bits = a_bits
        self.w_bits = w_bits
        
        self._use_w4a4 = _NUNCHAKU_AVAILABLE and a_bits == w_bits and a_bits in [2, 3, 4]
        if self._use_w4a4:
            if a_bits == 4:
                self._quant_fn_w4a4 = _symmetric_int4_quant
            elif a_bits == 3:
                self._quant_fn_w4a4 = _symmetric_int3_quant
            elif a_bits == 2:
                self._quant_fn_w4a4 = _symmetric_int2_quant
        
        self._w4a4_ready = False
        
    def forward(self, input):
        if (
            self._use_w4a4
            and self._w4a4_ready
            and input.dim() == 2
            and input.device.type == "cuda"
            and not torch.is_grad_enabled()
            and not self.training
        ):
            return self._forward_w4a4(input)
        return self._forward_fp(input)

    def _forward_fp(self, input: torch.Tensor) -> torch.Tensor:
        if self.if_hadamard:
            input = self.activation_hadamard(input)
        lora = None
        if self.if_lora:
            lora_module = getattr(self, "lora", None)
            if lora_module is not None:
                lora = lora_module(input)
        x = cast(torch.Tensor, self.activation_quantize(input))
        quantized_weights = cast(torch.Tensor, self.weight_quantize(self.weight))
        output = F.linear(x, quantized_weights, self.bias)
        if lora is not None:
            output = output + lora
        return output

    def prepare_for_inference(self):
        if not self._use_w4a4:
            warnings.warn(f"W{self.w_bits}A{self.n_bits} kernel 不可用或未启用, 回退到浮点前向。", RuntimeWarning)
            return
        if self._w4a4_ready:
            return
        if self.weight.device.type != "cuda":
            warnings.warn(f"W{self.w_bits}A{self.n_bits} (W4A4 Kernel) 推理需要CUDA设备。", RuntimeWarning)
            return
            
        weight_fp16 = self.weight.detach().to(torch.float16)
        qweight, wscales = self._quantize_weight_w4a4(weight_fp16)
        self._w4a4_qweight = qweight
        self._w4a4_wscales = wscales
        if self.bias is not None:
            bias_pad = torch.zeros((self._n_pad,), dtype=torch.float16, device=self.bias.device)
            bias_pad[: self.cout] = self.bias.detach().to(torch.float16)
            self._w4a4_bias = bias_pad
        else:
            self._w4a4_bias = None
        self._w4a4_ready = True

    def _quantize_weight_w4a4(self, weight_fp16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._use_w4a4:
            raise RuntimeError("W4A4兼容量化未启用")
        n_pad = self._n_pad
        k_pad = self._k_pad
        weight_padded = torch.zeros((n_pad, k_pad), dtype=torch.float16, device=weight_fp16.device)
        weight_padded[: self.cout, : self.cin] = weight_fp16
        groups = weight_padded.view(n_pad, k_pad // INT4_GROUP_SIZE, INT4_GROUP_SIZE)
        packed, scales = self._quant_fn_w4a4(groups) # 使用选择的量化函数
        return packed, scales

    def _quantize_activation_w4a4(self, input_fp16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        if not self._use_w4a4:
            raise RuntimeError("W4A4兼容量化未启用")
        batch, in_features = input_fp16.shape
        k_pad = self._k_pad
        if in_features > k_pad:
            raise ValueError("输入特征维度超过W4A4预设padding。")
        batch_pad = math.ceil(batch / W4A4_BATCH_PAD) * W4A4_BATCH_PAD
        act_pad = torch.zeros((batch_pad, k_pad), dtype=torch.float16, device=input_fp16.device)
        act_pad[:batch, :in_features] = input_fp16
        groups = act_pad.view(batch_pad, k_pad // INT4_GROUP_SIZE, INT4_GROUP_SIZE)
        packed, scales = self._quant_fn_w4a4(groups) # 使用选择的量化函数
        return packed, scales, batch_pad

    def _forward_w4a4(self, input: torch.Tensor) -> torch.Tensor:
        x = cast(torch.Tensor, input.to(torch.float16))
        if self.if_hadamard:
            x = self.activation_hadamard(x)
        lora_out = None
        if self.if_lora:
            lora_module = getattr(self, "lora", None)
            if lora_module is not None:
                lora_dtype = lora_module.A.weight.dtype
                lora_out = cast(torch.Tensor, lora_module(x.to(lora_dtype))).to(torch.float16)
        act_packed, ascales, batch_pad = self._quantize_activation_w4a4(x)
        out_pad = torch.zeros((batch_pad, self._n_pad), dtype=torch.float16, device=x.device)
        try:
            empty_lora_act = torch.empty((batch_pad, 0), dtype=torch.float32, device=x.device)
            empty_lora_up = torch.empty((self._n_pad, 0), dtype=torch.float16, device=x.device)
            svdq_gemm_w4a4_cuda(
                act=act_packed,
                wgt=self._w4a4_qweight,
                out=out_pad,
                ascales=ascales,
                wscales=self._w4a4_wscales,
                bias=self._w4a4_bias,
                lora_act_in=empty_lora_act,
                lora_up=empty_lora_up,
                lora_down=None,
                lora_act_out=None,
                smooth_factor=None,
                act_unsigned=False,
                lora_scales=[],
                fp4=False,
            )
        except Exception as exc:
            warnings.warn(f"W{self.w_bits}A{self.n_bits} (W4A4 Kernel) 调用失败, 自动回退到浮点实现: {exc}", RuntimeWarning)
            self._w4a4_ready = False
            return self._forward_fp(input)

        output = out_pad[: input.shape[0], : self.cout]
        if lora_out is not None:
            output = output + lora_out[: input.shape[0], : self.cout]
        return output.to(input.dtype)


def init_RobuQLinear_from_Linear(linear,n_bits=4,w_bits=4,if_hadamard=True,if_lora=True):

    robuq_linear = RobuQLinear(linear.in_features, linear.out_features, linear.bias is not None,linear.weight,if_lora=if_lora,if_hadamard=if_hadamard,n_bits=n_bits,w_bits=w_bits)
    if linear.bias is not None:
        robuq_linear.bias = linear.bias
    return robuq_linear