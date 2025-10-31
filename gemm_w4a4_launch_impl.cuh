#include "gemm_w4a4_launch.cuh"
#include <cstdint>

namespace nunchaku::kernels {

namespace {

constexpr int INT3_VALUES_PER_GROUP   = 64;
constexpr int INT3_BYTES_PER_GROUP    = (INT3_VALUES_PER_GROUP * 3) / 8; // 24
constexpr int INT4_BYTES_PER_GROUP    = INT3_VALUES_PER_GROUP / 2;       // 32
constexpr int INT3_SEGMENTS_PER_GROUP = INT3_VALUES_PER_GROUP / 8;

__device__ __forceinline__ int decode_int3(std::uint32_t bits, int idx) {
    int code = (bits >> (idx * 3)) & 0x7;
    return (code >= 4) ? (code - 8) : code;
}

__global__ void expand_int3_to_int4_kernel(const std::uint8_t *src,
                                           std::uint8_t *dst,
                                           int rows,
                                           int groups_per_row,
                                           int src_stride,
                                           int dst_stride) {
    int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total      = rows * groups_per_row;
    if (linear_idx >= total) {
        return;
    }

    int row   = linear_idx / groups_per_row;
    int group = linear_idx % groups_per_row;

    const std::uint8_t *in_ptr = src + row * src_stride + group * INT3_BYTES_PER_GROUP;
    std::uint8_t *out_ptr      = dst + row * dst_stride + group * INT4_BYTES_PER_GROUP;

#pragma unroll
    for (int seg = 0; seg < INT3_SEGMENTS_PER_GROUP; ++seg) {
        std::uint32_t chunk = std::uint32_t(in_ptr[seg * 3 + 0]) |
                              (std::uint32_t(in_ptr[seg * 3 + 1]) << 8) |
                              (std::uint32_t(in_ptr[seg * 3 + 2]) << 16);

#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            int idx0 = pair * 2;
            int idx1 = idx0 + 1;

            int val0 = decode_int3(chunk, idx0) << 1; // expand to int4 range [-8, 6]
            int val1 = decode_int3(chunk, idx1) << 1;

            std::uint8_t nibble0 = static_cast<std::uint8_t>(val0) & 0xF;
            std::uint8_t nibble1 = static_cast<std::uint8_t>(val1) & 0xF;

            out_ptr[seg * 4 + pair] = nibble0 | (nibble1 << 4);
        }
    }
}

inline void expand_int3_to_int4(const Tensor &src, Tensor &dst, int rows, int groups_per_row) {
    if (rows == 0 || groups_per_row == 0) {
        return;
    }
    assert(src.is_contiguous());
    assert(dst.is_contiguous());
    int threads = 128;
    int blocks  = ceilDiv(rows * groups_per_row, threads);

    const int src_stride = src.shape[-1];
    const int dst_stride = dst.shape[-1];

    assert(src_stride == groups_per_row * INT3_BYTES_PER_GROUP);
    assert(dst_stride == groups_per_row * INT4_BYTES_PER_GROUP);

    expand_int3_to_int4_kernel<<<blocks, threads, 0, getCurrentCUDAStream()>>>(
        reinterpret_cast<const std::uint8_t *>(src.data_ptr<std::int8_t>()),
        reinterpret_cast<std::uint8_t *>(dst.data_ptr<std::int8_t>()),
        rows,
        groups_per_row,
        src_stride,
        dst_stride);
    checkCUDA(cudaGetLastError());
}

} // namespace

#ifndef __INTELLISENSE__
template<typename Config, bool USE_FP4>
void GEMM_W4A4_Launch<Config, USE_FP4>::gemm_w4a4(
#else
template<>
void GEMM_W4A4_Launch<GEMMConfig_W4A4_FP16, false>::gemm_w4a4(
#endif
    Tensor act,            // packed act [M, K / 2]
    Tensor wgt,            // packed act [N, K / 2]
    Tensor out,            // linear     [M, N]
    Tensor qout,           // packed act [M, N / 2]
    Tensor ascales,        // packed as  [K / 64, M]
    Tensor wscales,        // packed ws  [K / 64, N]
    Tensor oscales,        // packed as  [N / 64, M]
    Tensor poolout,        // linear     [M / PoolSize, N]
    Tensor lora_act_in,    // packed lora_act [M, R]
    Tensor lora_up,        // packed lora_wgt [N, R]
    Tensor lora_down,      // packed lora_wgt [N, R]
    Tensor lora_act_out,   // packed lora_act [M, R]
    Tensor norm_q,         // linear     [HEAD_DIM]
    Tensor norm_k,         // linear     [HEAD_DIM]
    Tensor rotary_emb,     // linear     [M, HEAD_DIM / 2, 2, 2]
    Tensor bias,           // packed ws  [N]
    Tensor smooth_factor,  // packed ws  [N], for quantization of the next layer
    Tensor out_vk,         // linear     [B, num_heads, head_dim + 1, head_dim]
    Tensor out_linearattn, // linear     [B, (M), N / 3]
    bool act_unsigned,
    std::vector<float> lora_scales, // [R / 16]
    bool fuse_silu,
    bool fp4,
    float alpha,
    Tensor wcscales, // packed ws  [N]
    Tensor out_q,    // packed attention [B, H, M, D]
    Tensor out_k,    // packed attention [B, H, M, D]
    Tensor out_v,    // packed attention [B, H, M, D]
    int attn_tokens) {
#ifdef __INTELLISENSE__
    static constexpr bool USE_FP4 = false;
#endif
    assert(fp4 == USE_FP4);

    constexpr int GROUP_SIZE = GEMM::INSN_K;

    int groups = 0;
    if (ascales.valid()) {
        groups = ascales.shape[0];
    } else if (wscales.valid()) {
        groups = wscales.shape[0];
    }
    assert(groups > 0);
    int K_expected = groups * GROUP_SIZE;

    auto infer_bits = [&](const Tensor &tensor) -> int {
        if (!tensor.valid() || tensor.shape[-1] == 0) {
            return 4;
        }
        long long numerator = static_cast<long long>(tensor.shape[-1]) * 8;
        return static_cast<int>((numerator + K_expected - 1) / K_expected);
    };

    int M_rows = (act.shape[-1] == 0) ? 0 : act.numel() / act.shape[-1];
    int N_rows = wgt.shape[0];
    int act_bits = infer_bits(act);
    int wgt_bits = infer_bits(wgt);

    if (act_bits == 3) {
        assert(wgt_bits == 3);
        Tensor act_expanded = Tensor::allocate({M_rows, K_expected / 2}, Tensor::INT8, act.device());
        Tensor wgt_expanded = Tensor::allocate({N_rows, K_expected / 2}, Tensor::INT8, wgt.device());

        expand_int3_to_int4(act, act_expanded, M_rows, groups);
        expand_int3_to_int4(wgt, wgt_expanded, N_rows, groups);

        act = act_expanded;
        wgt = wgt_expanded;

        act_bits = 4;
        wgt_bits = 4;
    } else {
        assert(act_bits == 4);
        assert(wgt_bits == 4);
    }

    int M = (act.shape[-1] == 0) ? 0 : act.numel() / act.shape[-1];
    int N = wgt.shape[0];
    int K = act.shape[-1] * 2;
    assert(K == K_expected);
    assert(K == wgt.shape[-1] * 2);

    int actualM = 0;
    int actualN = 0;
    if (out.valid()) {
        actualM = out.numel() / out.shape[-1];
        actualN = out.shape[-1];

        assert(actualM <= M && M - actualM < GEMM::BLOCK_M);
        assert(actualN <= N && N - actualN < GEMM::BLOCK_N);
    }

    spdlog::trace("gemm_w4a4: M={} N={} K={}", M, N, K);
    spdlog::trace("act at {}", act.data_ptr());
    spdlog::trace("wgt at {}", wgt.data_ptr());
    spdlog::trace("ascales at {}", ascales.data_ptr());
    spdlog::trace("wscales at {}", wscales.data_ptr());
    if (bias.valid()) {
        spdlog::trace("bias at {}", bias.data_ptr());
    }

    int shmem = 0;

    auto launch = [&]<typename Epilogue>(Epilogue::Arguments args) {
        assert(M % GEMM::BLOCK_M == 0);
        assert(N % GEMM::BLOCK_N == 0);
        dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

        bool swapBlockMN = M > N * 2;
        if (swapBlockMN) {
            std::swap(grid.x, grid.y);
        }

        // test_sizeof<typename Epilogue::Arguments>();
        // std::apply([](auto ...args) {
        //     (test_sizeof<decltype(args)>(), ...);
        // }, args);

        // constexpr bool FP4_AVAILABLE = __CUDA_ARCH__ >= 1200;

        if constexpr (!USE_FP4) {
            dispatchBool(act_unsigned, [&]<bool ACT_UNSIGNED>() {
                auto func = invoke_kernel<typename GEMM::gemm_w4a4_kernel<Epilogue, ACT_UNSIGNED>,
                                          const packed_act_t *,
                                          const packed_wgt_t *,
                                          const packed_ascale_t *,
                                          const packed_wscale_t *,
                                          int,
                                          int,
                                          int,
                                          typename Epilogue::Arguments,
                                          bool,
                                          bool>;

                if (shmem >= 24 * 1024) {
                    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
                }

                assert(alpha == 1.0f);

                func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, shmem, getCurrentCUDAStream()>>>(
                    act.data_ptr<packed_act_t>(),
                    wgt.data_ptr<packed_wgt_t>(),
                    ascales.data_ptr<packed_ascale_t>(),
                    wscales.data_ptr<packed_wscale_t>(),
                    M,
                    N,
                    K,
                    args,
                    swapBlockMN,
                    false);
                checkCUDA(cudaGetLastError());
            });
            return;
        }

        if constexpr (USE_FP4) {
            dispatchBool(alpha != 1.0f, [&]<bool USE_ALPHA>() {
                assert(!act_unsigned);

                auto func = invoke_kernel<typename GEMM::gemm_w4a4_fp4_kernel<Epilogue, USE_ALPHA>,
                                          const packed_act_t *,
                                          const packed_wgt_t *,
                                          const packed_amscale_t *,
                                          const packed_wmscale_t *,
                                          float,
                                          int,
                                          int,
                                          int,
                                          typename Epilogue::Arguments,
                                          bool,
                                          bool>;

                if (shmem >= 24 * 1024) {
                    checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
                }

                assert(ascales.dtype() == Tensor::FP8_E4M3);
                assert(wscales.dtype() == Tensor::FP8_E4M3);

                func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, shmem, getCurrentCUDAStream()>>>(
                    act.data_ptr<packed_act_t>(),
                    wgt.data_ptr<packed_wgt_t>(),
                    ascales.data_ptr<packed_amscale_t>(),
                    wscales.data_ptr<packed_wmscale_t>(),
                    alpha,
                    M,
                    N,
                    K,
                    args,
                    swapBlockMN,
                    false);
                checkCUDA(cudaGetLastError());
            });

            return;
        }

        // if constexpr (USE_FP4 && !FP4_AVAILABLE) {
        //     throw std::runtime_error("FP4 kernel is not available");
        // }
    };

    auto launch_bias = [&]<typename NextEpilogue>(NextEpilogue::Arguments nextArgs) {
        assert(!bias.valid() || bias.numel() == N);
        assert(!wcscales.valid() || wcscales.numel() == N);

        dispatchBool(bias.valid(), [&]<bool USE_BIAS>() {
            dispatchBool(wcscales.valid(), [&]<bool USE_SCALE>() {
                using EpilogueBias = typename GEMM::EpilogueBias<USE_BIAS, USE_SCALE>;
                // append EpilgoueNop to workaround mismatched memory layout of std::tuple between device and host code
                // on Windows
                // ** sizeof(std::tuple<std::tuple<int>>) == 8 on device **
                using Epilogue =
                    typename GEMM::EpilogueCombination<EpilogueBias, NextEpilogue, typename GEMM::EpilogueNop>;
                return launch.template operator()<Epilogue>(
                    {typename EpilogueBias::Arguments{
                         .bias  = USE_BIAS ? bias.data_ptr<packed_wscale_t>() : nullptr,
                         .scale = USE_SCALE ? wcscales.data_ptr<packed_wscale_t>() : nullptr,
                     },
                     nextArgs,
                     {}});
            });
        });
    };
    // auto launch_bias = launch;

    auto launch_lora = [&]<typename NextEpilogue, typename MidEpilogue>(NextEpilogue::Arguments nextArgs,
                                                                        MidEpilogue::Arguments midArgs) {
        assert(lora_up.valid() == lora_act_in.valid());
        assert(lora_down.valid() == lora_act_out.valid());

        const int rank_up   = lora_up.valid() ? lora_up.shape[1] : 0;
        const int rank_down = lora_down.valid() ? lora_down.shape[1] : 0;

        if (rank_up == 0) {
            assert(rank_down == 0);
            return launch_bias.template operator()<typename GEMM::EpilogueCombination<MidEpilogue, NextEpilogue>>(
                {midArgs, nextArgs});
        }

        assert(rank_up % 16 == 0);

        assert(lora_up.shape[0] == N);
        // assert(lora_up.shape[1] == Lora::LORA_RANK);
        assert(lora_act_in.shape[0] == M);
        assert(lora_act_in.shape[1] == rank_up);

        using LoraUp  = Lora;
        using scale_t = typename LoraUp::scale_t;

        scale_t scales;
        if constexpr (scales.size() > 0) {
            for (size_t i = 0; i < scales.size(); i++) {
                scales[i] = i < lora_scales.size() ? lora_scales[i] : 0.0f;
            }
        }

        if (rank_down == 0) {
            using Epilogue = typename GEMM::EpilogueCombination<typename LoraUp::EpilogueLoraUp,
                                                                MidEpilogue,
                                                                NextEpilogue,
                                                                typename GEMM::EpilogueNop>;
            return launch_bias.template operator()<Epilogue>({typename LoraUp::EpilogueLoraUp::Arguments{
                                                                  .lora_act    = lora_act_in.data_ptr<float>(),
                                                                  .lora_wgt_up = lora_up.data_ptr<packed_fpsum_t>(),
                                                                  .rank        = rank_up,
                                                                  .scales      = scales,
                                                                  .alwaysfalse = false,
                                                              },
                                                              midArgs,
                                                              nextArgs,
                                                              {}});
        }

        // assert(rank_down == rank_up);
        assert(rank_down % 16 == 0);

        assert(lora_down.shape[0] == N);
        // assert(lora_down.shape[1] == Lora::LORA_RANK);
        assert(lora_act_out.shape[0] == M);
        assert(lora_act_out.shape[1] == rank_down);

        lora_act_out.zero_();

        // dispatchVal(rank_down, std::integer_sequence<int, 16, 32, 48, 64, 80>(), [&]<int RANK_DOWN>() {

        using LoraDown = LoraUp; // GEMM::Lora<RANK_DOWN>;
        using Epilogue = typename GEMM::EpilogueCombination<typename LoraUp::EpilogueLoraUp,
                                                            MidEpilogue,
                                                            typename LoraDown::EpilogueLoraDown,
                                                            NextEpilogue,
                                                            typename GEMM::EpilogueNop>;
        return launch_bias.template operator()<Epilogue>({typename LoraUp::EpilogueLoraUp::Arguments{
                                                              .lora_act    = lora_act_in.data_ptr<float>(),
                                                              .lora_wgt_up = lora_up.data_ptr<packed_fpsum_t>(),
                                                              .rank        = rank_up,
                                                              .scales      = scales,
                                                              .alwaysfalse = false,
                                                          },
                                                          midArgs,
                                                          typename LoraDown::EpilogueLoraDown::Arguments{
                                                              .lora_wgt_down = lora_down.data_ptr<packed_fpsum_t>(),
                                                              .lora_act      = lora_act_out.data_ptr<float>(),
                                                              .rank          = rank_down,
                                                              .alwaysfalse   = false,
                                                          },
                                                          nextArgs,
                                                          {}});

        // });
    };

    if (qout.valid() && oscales.valid()) {

        // dispatchBool(qout_unsigned, [&]<bool USE_UNSIGNED>() {

        static constexpr float SHIFT_GELU = 0.171875f;

        constexpr bool USE_UNSIGNED = !USE_FP4;
        using EpilogueQuantize      = typename GEMM::EpilogueQuantize<false, USE_UNSIGNED, USE_FP4>;
        auto argsQuantize =
            typename EpilogueQuantize::Arguments{.qout    = qout.data_ptr<packed_act_t>(),
                                                 .oscales = oscales.data_ptr<typename EpilogueQuantize::oscales_t>(),
                                                 .shift_value   = USE_FP4 ? 0.0f : SHIFT_GELU,
                                                 .smooth_factor = smooth_factor.data_ptr<packed_wscale_t>()};

        // TODO: check if gelu is needed
        if (out.valid()) {
            launch_lora.template
            operator()<typename GEMM::EpilogueCombination<typename GEMM::EpilogueDefault, EpilogueQuantize>,
                       typename Epilogues::EpilogueGelu>({typename GEMM::EpilogueDefault::Arguments{
                                                              .out     = out.data_ptr<half_t>(),
                                                              .actualM = actualM,
                                                              .actualN = actualN,
                                                          },
                                                          argsQuantize},
                                                         {});
        } else {
            launch_lora.template operator()<EpilogueQuantize, typename Epilogues::EpilogueGelu>(argsQuantize, {});
        }

    } else if (out_linearattn.valid()) {

        assert(out_vk.valid());

        using Epilogue = typename Epilogues::EpilogueLiteLA;

        assert(out_vk.dtype() == Tensor::FP32);
        assert(out_vk.ndims() == 4);
        assert(out_vk.shape[2] == Epilogue::LITELA_HEAD_DIM + 1);
        assert(out_vk.shape[3] == Epilogue::LITELA_HEAD_DIM);
        assert(out_vk.shape[1] * Epilogue::LITELA_HEAD_DIM * 3 == N);
        int batch_size = out_vk.shape[0];
        int num_heads  = out_vk.shape[1];

        assert(isTypeMatch<half_t>(out_linearattn.dtype()));
        assert(out_linearattn.ndims() == 3);
        assert(out_linearattn.shape[0] == batch_size);
        assert(out_linearattn.shape[2] * 3 == N);
        int num_tokens = out_linearattn.shape[1];

        assert(num_tokens % GEMM::BLOCK_M == 0);
        int num_blocks_per_batch = ceilDiv(num_tokens, GEMM::BLOCK_M);

        shmem = std::max(shmem, Epilogue::SHMEM_SIZE);

        out_vk.zero_();

        launch_lora.template operator()<Epilogue, typename GEMM::EpilogueNop>(
            typename Epilogue::Arguments{
                .out_q                = out_linearattn.data_ptr<half_t>(),
                .out_vk               = out_vk.data_ptr<float>(),
                .num_blocks_per_batch = num_blocks_per_batch,
                .actualM              = M,
            },
            {});

    } else if (rotary_emb.valid()) {
        assert(norm_q.valid());
        assert(norm_k.valid());
        // assert(isTypeMatch<half_t>(rotary_emb.scalar_type()));
        assert(rotary_emb.scalar_type() == Tensor::FP32);
        assert(rotary_emb.ndims() == 3);
        assert(rotary_emb.shape[0] * rotary_emb.shape[1] == M);
        assert(rotary_emb.shape[2] == Epilogues::EpilogueRMSNormRope::HEAD_DIM);

        // assert(rotary_emb.numel() == M * GEMM::EpilogueQKVProj::HEAD_DIM / 2 *
        // GEMM::EpilogueQKVProj::ROTARY_EMB_NUM_ELEMENTS); launch_lora.template operator()<typename
        // GEMM::EpilogueQKVProj, typename GEMM::EpilogueNop>(typename GEMM::EpilogueQKVProj::Arguments{
        //     .out = out.data_ptr<half_t>(),
        //     .actualM = actualM,
        //     .actualN = actualN,
        //     .pool_out = poolout.valid() ? poolout.data_ptr<half_t>() : nullptr,
        //     .rotary_emb = rotary_emb.data_ptr<float>(),
        //     .rmsnorm_weight_q = norm_q.data_ptr<half_t>(),
        //     .rmsnorm_weight_k = norm_k.data_ptr<half_t>(),
        //     .epsilon = 1e-6,
        // }, {});

        using EpilogueRope = typename Epilogues::EpilogueRMSNormRope;
        auto argsRope      = typename Epilogues::EpilogueRMSNormRope::Arguments{
                 .rotary_emb       = rotary_emb.data_ptr<typename EpilogueRope::packed_rotemb_t>(),
                 .rmsnorm_weight_q = norm_q.data_ptr<half_t>(),
                 .rmsnorm_weight_k = norm_k.data_ptr<half_t>(),
                 .epsilon          = 1e-6,
        };

        if (out_q.valid()) {
            launch_lora.template
            operator()<typename GEMM::EpilogueCombination<EpilogueRope, typename Epilogues::EpiloguePackQKV>,
                       typename GEMM::EpilogueNop>(
                {argsRope,
                 typename Epilogues::EpiloguePackQKV::Arguments{
                     .out_q        = out_q.data_ptr<typename Epilogues::EpiloguePackQKV::packed_qkv_t>(),
                     .out_k        = out_k.data_ptr<typename Epilogues::EpiloguePackQKV::packed_qkv_t>(),
                     .out_v        = out_v.data_ptr<typename Epilogues::EpiloguePackQKV::packed_qkv_t>(),
                     .actualM      = attn_tokens,
                     .strideHead_q = int(out_q.stride(1) * out_q.scalar_size() /
                                         sizeof(typename Epilogues::EpiloguePackQKV::packed_qkv_t)),
                     .strideHead_k = int(out_k.stride(1) * out_k.scalar_size() /
                                         sizeof(typename Epilogues::EpiloguePackQKV::packed_qkv_t)),
                     .strideHead_v = int(out_v.stride(1) * out_v.scalar_size() /
                                         sizeof(typename Epilogues::EpiloguePackQKV::packed_qkv_t)),
                 }},
                {});
        } else {
            launch_lora
                .template operator()<typename GEMM::EpilogueCombination<EpilogueRope, typename GEMM::EpilogueDefault>,
                                     typename GEMM::EpilogueNop>({argsRope,
                                                                  typename GEMM::EpilogueDefault::Arguments{
                                                                      .out     = out.data_ptr<half_t>(),
                                                                      .actualM = actualM,
                                                                      .actualN = actualN,
                                                                  }},
                                                                 {});
        }

    } else if (out.valid()) {

        using Epilogue = typename GEMM::EpilogueDefault;
        typename Epilogue::Arguments args{
            .out     = out.data_ptr<half_t>(),
            .actualM = actualM,
            .actualN = actualN,
        };

        if (fuse_silu) {
            launch_lora.template operator()<Epilogue, typename GEMM::EpilogueSilu>(args, {});
        } else {
            launch_lora.template operator()<Epilogue, typename GEMM::EpilogueNop>(args, {});
        }
    } else {
        assert(false);
    }
}

template<typename Config, bool USE_FP4>
void GEMM_W4A4_Launch<Config, USE_FP4>::linearattn_vk_mul_q(Tensor q, Tensor vk) {
    using Epilogue = typename Epilogues::EpilogueLiteLA;

    int batch_size = vk.shape[0];
    int num_heads  = vk.shape[1];
    int num_tokens = q.shape[1];

    assert(isTypeMatch<half_t>(q.scalar_type()));
    assert(vk.scalar_type() == Tensor::FP32);

    int BLOCK_SIZE;
    if (num_tokens % 256 == 0) {
        BLOCK_SIZE = 256;
    } else {
        BLOCK_SIZE = 128;
    }

    invoke_kernel<typename Epilogue::vk_mul_q_kernel>
        <<<dim3(ceilDiv(num_tokens, BLOCK_SIZE), num_heads, batch_size), BLOCK_SIZE, 0, getCurrentCUDAStream()>>>(
            q.data_ptr<half_t>(), vk.data_ptr<float>(), 1e-6f, num_tokens);
    checkCUDA(cudaGetLastError());
}

template<typename Config, bool USE_FP4>
void GEMM_W4A4_Launch<Config, USE_FP4>::quantize_w4a4_act_fuse_lora(Tensor input,
                                                                    Tensor output,
                                                                    Tensor oscales,
                                                                    Tensor lora_down,
                                                                    Tensor lora_act_out,
                                                                    Tensor smooth,
                                                                    bool fuse_glu,
                                                                    bool fp4) {
    const int actualM = input.numel() / input.shape[-1];
    const int actualN = input.shape[-1];

    const int M = ceilDiv(actualM, GEMM::BLOCK_M) * GEMM::BLOCK_M;
    const int N = ceilDiv(actualN / (fuse_glu ? 2 : 1), GEMM::BLOCK_N) * GEMM::BLOCK_N;

    assert(output.dtype() == Tensor::INT8);
    assert(output.numel() / output.shape[-1] == M);
    assert(output.shape[-1] == N / 2);

    // assert(oscales.dtype() == Tensor::FP16);
    if (fp4) {
        assert(oscales.dtype() == Tensor::FP8_E4M3);
        assert(oscales.numel() == M * N / GEMM::WARP_K * 4);
    } else {
        assert(isTypeMatch<half_t>(oscales.dtype()));
        assert(oscales.numel() == M * N / GEMM::WARP_K);
    }

    const int rank = lora_down.shape[1];

    assert(rank % 16 == 0);

    assert(lora_down.shape[0] == N);
    // assert(lora_down.shape[1] == Lora::LORA_RANK);
    assert(lora_act_out.shape[0] == M);
    assert(lora_act_out.shape[1] == rank);

    lora_act_out.zero_();

    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    // dispatchVal(rank, LoraRanks(), [&]<int RANK>() {
    dispatchBool(fuse_glu, [&]<bool FUSE_GLU>() {
        // using Lora = typename GEMM::Lora<RANK>;
        using kernel = typename GEMM::quantize_w4a4_fuse_lora_kernel<FUSE_GLU, USE_FP4>;

        auto func = invoke_kernel<kernel, typename kernel::Arguments>;

        checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel::SHMEM_SIZE));

        // log(std::format("quantize_w4a4_act_fuse_lora M={} N={} input={} output={} (size={} numel={})", M, N,
        // input.data_ptr(), output.data_ptr(), output.buffer->getSize(), output.numel()));

        func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, kernel::SHMEM_SIZE, getCurrentCUDAStream()>>>(
            typename kernel::Arguments{
                .input         = input.data_ptr<half_t>(),
                .smooth_factor = smooth.valid() ? smooth.data_ptr<packed_wscale_t>() : nullptr,
                .output        = output.data_ptr<packed_act_t>(),
                .oscales       = oscales.data_ptr<typename kernel::oscales_t>(),
                .lora_wgt_down = lora_down.data_ptr<packed_fpsum_t>(),
                .lora_act      = lora_act_out.data_ptr<float>(),
                .lora_rank     = rank,
                .M             = M,
                .N             = N,
                .actualM       = actualM,
                .actualN       = actualN,
                .alwaysfalse   = false,
            });
        checkCUDA(cudaGetLastError());
    });
    // });
}

template<typename Config, bool USE_FP4>
void GEMM_W4A4_Launch<Config, USE_FP4>::quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales) {
    if constexpr (USE_FP4) {
        assert(false); // not implemented
        return;
    }

    int M = input.numel() / input.shape[-1];
    int K = input.shape[-1];

    assert(output.dtype() == Tensor::INT8);
    assert(output.numel() / output.shape[-1] == M);
    assert(output.shape[-1] == K / 2);

    // assert(oscales.dtype() == Tensor::FP16);
    assert(isTypeMatch<half_t>(oscales.dtype()));
    assert(oscales.numel() == M * K / GEMM::WARP_K);

    dim3 grid(M / GEMM::WARP_M, K / GEMM::WARP_K);
    invoke_kernel<typename GEMM::quantize_w4a4_act_kernel><<<grid, GEMM::WARP_SIZE, 0, getCurrentCUDAStream()>>>(
        input.data_ptr<half_t>(), output.data_ptr<packed_act_t>(), oscales.data_ptr<packed_ascale_t>(), K);
    checkCUDA(cudaGetLastError());
}

template<typename Config, bool USE_FP4>
void GEMM_W4A4_Launch<Config, USE_FP4>::quantize_w4a4_wgt(Tensor input, Tensor output, Tensor oscales) {
    if constexpr (USE_FP4) {
        assert(false);
        return;
    }

    int N = input.numel() / input.shape[-1];
    int K = input.shape[-1];

    assert(output.dtype() == Tensor::INT8);
    assert(output.ndims() == 2);
    assert(output.shape[0] == N);
    assert(output.shape[1] == K / 2);

    assert(isTypeMatch<half_t>(oscales.dtype()));
    // assert(oscales.dtype() == Tensor::FP16);
    assert(oscales.numel() == N * K / GEMM::WARP_K);

    dim3 grid(N / GEMM::WARP_N, K / GEMM::WARP_K);
    invoke_kernel<typename GEMM::quantize_w4a4_wgt_kernel><<<grid, GEMM::WARP_SIZE, 0, getCurrentCUDAStream()>>>(
        input.data_ptr<half_t>(), output.data_ptr<packed_wgt_t>(), oscales.data_ptr<packed_wscale_t>(), K);
    checkCUDA(cudaGetLastError());
}

}; // namespace nunchaku::kernels
