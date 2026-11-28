#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include "cu_mir_cache.h"
#include "cuda_driver.h"
#include "detail/cuda_check.h"
#include "gemm_mma.h"
#include "ops_fa_mir.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace brotensor {

void* cuda_current_stream();

namespace detail::cuda {

namespace {

constexpr float kPScale = 16384.0f;
constexpr std::size_t kBwdScoreBudget = std::size_t(8) << 20;

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

std::uint32_t grid_for(long long n, int block) {
    long long b = (n + block - 1) / block;
    if (b < 1) b = 1;
    return static_cast<std::uint32_t>(b);
}

int round8(int n) { return (n + 7) & ~7; }

template <typename T>
bool run_f32_gemm(const void* a, const void* b, float* c, int m, int n, int k, CUstream stream) {
    return mma_gemm::launch_f32out(static_cast<const T*>(a), static_cast<const T*>(b), c, m, n, k, stream);
}

bool launch_f32_gemm_dt(Dtype dt, const void* a, const void* b, float* c, int m, int n, int k, CUstream stream) {
    if (dt == Dtype::BF16) return run_f32_gemm<Bf16>(a, b, c, m, n, k, stream);
    return run_f32_gemm<Fp16>(a, b, c, m, n, k, stream);
}

bool tc_backward_dt(Dtype dt, const char* Q, const char* K, const char* V, const char* dO,
                    const float* mask, bool causal, char* dQ, char* dK, char* dV,
                    int Lq, int Lk, int ld, int nh, int hd, CUstream stream) {
    const int hdp = round8(hd), Lk_pad = round8(Lk);
    int rows = Lq;
    if (static_cast<std::size_t>(Lq) * Lk_pad > kBwdScoreBudget) {
        rows = static_cast<int>(kBwdScoreBudget / Lk_pad) / 128 * 128;
        if (rows < 128) rows = 128;
        if (rows > Lq) rows = Lq;
    }
    const int rows_pad = round8(rows);

    const Device cur_dev = Device::cuda(current_device());
    auto buf = [&](std::size_t n, Dtype d) {
        return Tensor::empty_on(cur_dev, static_cast<int>(n), 1, d);
    };
    const std::size_t kd = static_cast<std::size_t>(Lk_pad) * hdp;
    const std::size_t qd = static_cast<std::size_t>(rows_pad) * hdp;
    const std::size_t sc = static_cast<std::size_t>(rows) * Lk_pad;
    const std::size_t tc = static_cast<std::size_t>(Lk_pad) * rows_pad;

    Tensor Kh = buf(kd, dt), Vh = buf(kd, dt), KhT = buf(2 * kd, dt);
    Tensor Qh = buf(qd, dt), dOh = buf(qd, dt), QhT = buf(2 * qd, dt), dOhT = buf(qd, dt);
    Tensor Sb = buf(sc, Dtype::FP32), dPb = buf(sc, Dtype::FP32), dSb = buf(2 * sc, dt);
    Tensor dSTb = buf(2 * tc, dt), PTb = buf(tc, dt), stats = buf(static_cast<std::size_t>(rows) * 3, Dtype::FP32);
    Tensor dq32 = buf(qd, Dtype::FP32), dk32 = buf(kd, Dtype::FP32), dv32 = buf(kd, Dtype::FP32);
    Tensor tmp = buf(2 * kd, Dtype::FP32), bound = buf(1, Dtype::FP32);

    float* s = static_cast<float*>(Sb.data);
    float* dp = static_cast<float*>(dPb.data);
    float* tk = static_cast<float*>(tmp.data);
    float* tv = tk + kd;

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const float scale_log2 = scale * 1.4426950408889634f;
    constexpr int CP = 256;
    const std::uint8_t cm_flags = static_cast<std::uint8_t>((causal ? 1u : 0u) | (mask ? 2u : 0u));

    CUfunction gather_fn = mir_function({MirKind::FaAux, 1u, dt, dt, 1u});
    CUfunction gather_t0 = mir_function({MirKind::FaAux, 2u, dt, dt, 1u, Dtype::FP32, 0u});
    CUfunction gather_t1 = mir_function({MirKind::FaAux, 2u, dt, dt, 1u, Dtype::FP32, 1u});
    CUfunction scatter_s = mir_function({MirKind::FaAux, 5u, dt, dt, 1u, Dtype::FP32, 1u});
    CUfunction scatter_n = mir_function({MirKind::FaAux, 5u, dt, dt, 1u, Dtype::FP32, 0u});
    CUfunction acc_s_1 = mir_function({MirKind::FaAux, 6u, Dtype::FP32, Dtype::FP32, 1u, Dtype::FP32, 3u});
    CUfunction acc_s_n = mir_function({MirKind::FaAux, 6u, Dtype::FP32, Dtype::FP32, 1u, Dtype::FP32, 1u});
    CUfunction acc_c_1 = mir_function({MirKind::FaAux, 6u, Dtype::FP32, Dtype::FP32, 1u, Dtype::FP32, 2u});
    CUfunction acc_c_n = mir_function({MirKind::FaAux, 6u, Dtype::FP32, Dtype::FP32, 1u, Dtype::FP32, 0u});
    CUfunction rstat_fn = mir_function({MirKind::FaAux, 7u, Dtype::FP32, Dtype::FP32, 1u, Dtype::FP32, cm_flags});
    CUfunction ds_fn = mir_function({MirKind::FaAux, 8u, dt, dt, 1u, Dtype::FP32, cm_flags});

    for (int h = 0; h < nh; ++h) {
        const int c0 = h * hd;
        mir_dispatch_args(gather_fn, grid_for(static_cast<long long>(kd), CP), CP,
                          std::array<std::uintptr_t, 2>{addr(K), addr(Kh.data)},
                          std::array<std::int32_t, 6>{ld, c0, Lk, hd, Lk_pad, hdp}, {});
        mir_dispatch_args(gather_fn, grid_for(static_cast<long long>(kd), CP), CP,
                          std::array<std::uintptr_t, 2>{addr(V), addr(Vh.data)},
                          std::array<std::int32_t, 6>{ld, c0, Lk, hd, Lk_pad, hdp}, {});
        mir_dispatch_args(gather_t1, grid_for(static_cast<long long>(kd), CP), CP,
                          std::array<std::uintptr_t, 2>{addr(K), addr(KhT.data)},
                          std::array<std::int32_t, 7>{ld, c0, Lk, hd, Lk_pad, hdp, 2 * Lk_pad}, {});

        for (int q0 = 0; q0 < Lq; q0 += rows) {
            const int m = Lq - q0 < rows ? Lq - q0 : rows;
            const int m_pad = round8(m);
            const std::size_t mo_bytes = static_cast<std::size_t>(q0) * ld * 2u;
            const long long mq = static_cast<long long>(m) * hdp;
            const long long mqT = static_cast<long long>(m_pad) * hdp;

            mir_dispatch_args(gather_fn, grid_for(mq, CP), CP,
                              std::array<std::uintptr_t, 2>{addr(Q + mo_bytes), addr(Qh.data)},
                              std::array<std::int32_t, 6>{ld, c0, m, hd, m, hdp}, {});
            mir_dispatch_args(gather_fn, grid_for(mq, CP), CP,
                              std::array<std::uintptr_t, 2>{addr(dO + mo_bytes), addr(dOh.data)},
                              std::array<std::int32_t, 6>{ld, c0, m, hd, m, hdp}, {});
            mir_dispatch_args(gather_t1, grid_for(mqT, CP), CP,
                              std::array<std::uintptr_t, 2>{addr(Q + mo_bytes), addr(QhT.data)},
                              std::array<std::int32_t, 7>{ld, c0, m, hd, m_pad, hdp, 2 * m_pad}, {});
            mir_dispatch_args(gather_t0, grid_for(mqT, CP), CP,
                              std::array<std::uintptr_t, 2>{addr(dO + mo_bytes), addr(dOhT.data)},
                              std::array<std::int32_t, 7>{ld, c0, m, hd, m_pad, hdp, m_pad}, {});

            if (!launch_f32_gemm_dt(dt, Qh.data, Kh.data, s, m, Lk_pad, hdp, stream)) return false;
            launch_f32_gemm_dt(dt, dOh.data, Vh.data, dp, m, Lk_pad, hdp, stream);
            BROTENSOR_CUDA_CHECK(memset_zero_async(bound.data, sizeof(unsigned int), stream));

            mir_dispatch_args(rstat_fn, static_cast<std::uint32_t>(m), 256u,
                              std::array<std::uintptr_t, 5>{addr(s), addr(dp), addr(mask), addr(stats.data), addr(bound.data)},
                              std::array<std::int32_t, 3>{Lk_pad, Lk, q0},
                              std::array<float, 2>{scale_log2, scale});
            mir_dispatch_args_3d(ds_fn, static_cast<std::uint32_t>((Lk_pad + 31) / 32),
                                 static_cast<std::uint32_t>((m_pad + 31) / 32), 1u, 256u,
                                 std::array<std::uintptr_t, 8>{addr(s), addr(dp), addr(stats.data), addr(bound.data),
                                                               addr(mask), addr(dSb.data), addr(dSTb.data), addr(PTb.data)},
                                 std::array<std::int32_t, 6>{Lk_pad, m, m_pad, Lk, Lk_pad, q0},
                                 std::array<float, 2>{scale_log2, scale});

            launch_f32_gemm_dt(dt, dSb.data, KhT.data, static_cast<float*>(dq32.data), m, hdp, 2 * Lk_pad, stream);
            mir_dispatch_args(scatter_s, grid_for(static_cast<long long>(m) * hd, CP), CP,
                              std::array<std::uintptr_t, 3>{addr(dq32.data), addr(dQ + mo_bytes), addr(bound.data)},
                              std::array<std::int32_t, 5>{hdp, m, hd, ld, c0}, {});

            const bool first = (q0 == 0);
            launch_f32_gemm_dt(dt, dSTb.data, QhT.data, tk, Lk_pad, hdp, 2 * m_pad, stream);
            launch_f32_gemm_dt(dt, PTb.data, dOhT.data, tv, Lk_pad, hdp, m_pad, stream);

            mir_dispatch_args(first ? acc_s_1 : acc_s_n, grid_for(static_cast<long long>(kd), CP), CP,
                              std::array<std::uintptr_t, 3>{addr(dk32.data), addr(tk), addr(bound.data)},
                              std::array<std::int32_t, 1>{static_cast<std::int32_t>(kd)},
                              std::array<float, 1>{1.0f});
            mir_dispatch_args(first ? acc_c_1 : acc_c_n, grid_for(static_cast<long long>(kd), CP), CP,
                              std::array<std::uintptr_t, 3>{addr(dv32.data), addr(tv), 0u},
                              std::array<std::int32_t, 1>{static_cast<std::int32_t>(kd)},
                              std::array<float, 1>{1.0f / kPScale});
        }

        const long long kh_n = static_cast<long long>(Lk) * hd;
        mir_dispatch_args(scatter_n, grid_for(kh_n, CP), CP,
                          std::array<std::uintptr_t, 3>{addr(dk32.data), addr(dK), 0u},
                          std::array<std::int32_t, 5>{hdp, Lk, hd, ld, c0}, {});
        mir_dispatch_args(scatter_n, grid_for(kh_n, CP), CP,
                          std::array<std::uintptr_t, 3>{addr(dv32.data), addr(dV), 0u},
                          std::array<std::int32_t, 5>{hdp, Lk, hd, ld, c0}, {});
    }
    return true;
}

bool tc_block(Dtype dt, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& dO, const float* mask,
              bool causal, Tensor& dQ, Tensor& dK, Tensor& dV, std::size_t q_row, std::size_t k_row, int Lq, int Lk,
              int D, int nh, int hd, CUstream stream) {
    if (dt != Dtype::BF16 && dt != Dtype::FP16) return false;
    const std::size_t qo = q_row * D * 2u;
    const std::size_t ko = k_row * D * 2u;
    return tc_backward_dt(dt,
                          static_cast<const char*>(Q.data) + qo,
                          static_cast<const char*>(K.data) + ko,
                          static_cast<const char*>(V.data) + ko,
                          static_cast<const char*>(dO.data) + qo,
                          mask, causal,
                          static_cast<char*>(dQ.data) + qo,
                          static_cast<char*>(dK.data) + ko,
                          static_cast<char*>(dV.data) + ko,
                          Lq, Lk, D, nh, hd, stream);
}

bool prefer_rows(const std::vector<std::int32_t>& cq, const std::vector<std::int32_t>& ck, int nh, int hd, bool causal) {
    if (const char* e = std::getenv("BROTENSOR_ATTN_BWD_PATH")) {
        if (e[0] == 'r') return true;
        if (e[0] == 't') return false;
    }
    const int npl = (hd + 31) / 32;
    if (npl > 8) return false;
    double rows_pairs = 0.0, tc_pairs = 0.0, blocks = 0.0;
    int max_q = 0, max_k = 0;
    for (std::size_t b = 0; b + 1 < cq.size(); ++b) {
        const double lq = cq[b + 1] - cq[b], lk = ck[b + 1] - ck[b];
        if (lq <= 0.0 || lk <= 0.0) continue;
        tc_pairs += lq * lk;
        rows_pairs += causal ? lq * (lq + 1.0) * 0.5 : lq * lk;
        blocks += 1.0;
        max_q = std::max(max_q, cq[b + 1] - cq[b]);
        max_k = std::max(max_k, ck[b + 1] - ck[b]);
    }
    const double c_rows = 0.26 + 0.1 * std::max(0, npl - 2);
    const double rows_ms = rows_pairs * nh * c_rows * 1e-6 + (2.0 * max_k + max_q) * 0.2e-3;
    const double tc_ms = std::max(tc_pairs * nh * (0.055 + 1e-4 * hd) * 1e-6, blocks * nh * 0.2);
    return rows_ms <= tc_ms;
}

}  // namespace

void flash_attention_backward(const Tensor& Q, const Tensor& K, const Tensor& V,
                              const Tensor& O, const Tensor& dO, const float* d_mask,
                              int num_heads, bool causal, Tensor& dQ, Tensor& dK, Tensor& dV) {
    (void)O;
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO must be FP16 or BF16");
    }
    if (K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_backward: Q, K, V, dO dtype must match");
    }
    const int Lq = Q.rows, Lk = K.rows, D = Q.cols;
    if (K.cols != D || V.cols != D || V.rows != Lk) {
        throw std::runtime_error("flash_attention_backward: Q/K/V shape mismatch");
    }
    if (dO.rows != Lq || dO.cols != D) {
        throw std::runtime_error("flash_attention_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || D % num_heads != 0) {
        throw std::runtime_error("flash_attention_backward: num_heads must divide D");
    }
    if (causal && Lq != Lk) {
        throw std::runtime_error("flash_attention_backward: causal requires Lq == Lk");
    }
    const int hd = D / num_heads;

    if (dQ.rows != Lq || dQ.cols != D || dQ.dtype != dt) dQ.resize(Lq, D, dt);
    if (dK.rows != Lk || dK.cols != D || dK.dtype != dt) dK.resize(Lk, D, dt);
    if (dV.rows != Lk || dV.cols != D || dV.dtype != dt) dV.resize(Lk, D, dt);
    if (Lq == 0 || Lk == 0 || D == 0) {
        dQ.zero();
        dK.zero();
        dV.zero();
        return;
    }

    CUstream stream = reinterpret_cast<CUstream>(cuda_current_stream());
    if (!prefer_rows({0, Lq}, {0, Lk}, num_heads, hd, causal) &&
        tc_block(dt, Q, K, V, dO, d_mask, causal, dQ, dK, dV, 0, 0, Lq, Lk, D, num_heads, hd, stream)) {
        return;
    }
    dispatch_fa_bwd(kFaModeRows, dt, Q.device,
                    Q.data, K.data, V.data, dO.data,
                    d_mask, nullptr, nullptr,
                    dQ.data, dK.data, dV.data,
                    Lq, Lk, D, D, D, D,
                    num_heads, hd, 0, causal, 0);
}

void flash_attention_varlen_backward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                     const Tensor& O, const Tensor& dO,
                                     const std::int32_t* cu_seqlens_q, const std::int32_t* cu_seqlens_k,
                                     int batch_size, int max_seqlen_q, int max_seqlen_k,
                                     int num_heads, int head_dim, bool causal,
                                     Tensor& dQ, Tensor& dK, Tensor& dV) {
    (void)O;
    const Dtype dt = Q.dtype;
    if (dt != Dtype::FP16 && dt != Dtype::BF16 && dt != Dtype::FP32) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO must be FP16, BF16, or FP32");
    }
    if (K.dtype != dt || V.dtype != dt || dO.dtype != dt) {
        throw std::runtime_error("flash_attention_varlen_backward: Q, K, V, dO dtype must match");
    }
    const int total_q = Q.rows, total_k = K.rows, D = num_heads * head_dim;
    if (Q.cols != D || K.cols != D || V.cols != D || V.rows != total_k) {
        throw std::runtime_error("flash_attention_varlen_backward: shape mismatch");
    }
    if (dO.rows != total_q || dO.cols != D) {
        throw std::runtime_error("flash_attention_varlen_backward: dO shape mismatch");
    }
    if (num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("flash_attention_varlen_backward: num_heads/head_dim must be positive");
    }
    if (batch_size < 0) {
        throw std::runtime_error("flash_attention_varlen_backward: batch_size must be non-negative");
    }
    if (batch_size > 0 && (!cu_seqlens_q || !cu_seqlens_k)) {
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q/k required when batch_size > 0");
    }
    if (max_seqlen_q < 0 || max_seqlen_k < 0) {
        throw std::runtime_error("flash_attention_varlen_backward: max_seqlen_q/k must be non-negative");
    }

    if (dQ.rows != total_q || dQ.cols != D || dQ.dtype != dt) dQ.resize(total_q, D, dt);
    if (dK.rows != total_k || dK.cols != D || dK.dtype != dt) dK.resize(total_k, D, dt);
    if (dV.rows != total_k || dV.cols != D || dV.dtype != dt) dV.resize(total_k, D, dt);
    dQ.zero();
    dK.zero();
    dV.zero();

    if (D == 0 || batch_size == 0) return;
    if (total_q == 0 && total_k == 0) return;

    CUstream stream = reinterpret_cast<CUstream>(cuda_current_stream());
    std::vector<std::int32_t> cq(batch_size + 1), ck(batch_size + 1);
    BROTENSOR_CUDA_CHECK(memcpy_d2h_async(cq.data(), cu_seqlens_q, sizeof(std::int32_t) * (batch_size + 1), stream));
    BROTENSOR_CUDA_CHECK(memcpy_d2h_async(ck.data(), cu_seqlens_k, sizeof(std::int32_t) * (batch_size + 1), stream));
    BROTENSOR_CUDA_CHECK(drv::cuStreamSynchronize(stream));

    if (cq[0] != 0) throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[0] must be 0");
    if (ck[0] != 0) throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[0] must be 0");
    if (cq[batch_size] != total_q)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_q[B] != total_tokens_q");
    if (ck[batch_size] != total_k)
        throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens_k[B] != total_tokens_k");

    bool any = false;
    for (int b = 0; b < batch_size; ++b) {
        const int Lq_b = cq[b + 1] - cq[b], Lk_b = ck[b + 1] - ck[b];
        if (Lq_b < 0 || Lk_b < 0)
            throw std::runtime_error("flash_attention_varlen_backward: cu_seqlens must be non-decreasing");
        if (causal && Lq_b != Lk_b)
            throw std::runtime_error("flash_attention_varlen_backward: causal requires per-sequence Lq == Lk");
        any = any || (Lq_b > 0 && Lk_b > 0);
    }
    if (!any) return;

    if (!prefer_rows(cq, ck, num_heads, head_dim, causal)) {
        bool ran = false, refused = false;
        for (int b = 0; b < batch_size && !refused; ++b) {
            const int Lq = cq[b + 1] - cq[b], Lk = ck[b + 1] - ck[b];
            if (Lq == 0 || Lk == 0) continue;
            if (tc_block(dt, Q, K, V, dO, nullptr, causal, dQ, dK, dV, static_cast<std::size_t>(cq[b]),
                         static_cast<std::size_t>(ck[b]), Lq, Lk, D, num_heads, head_dim, stream))
                ran = true;
            else
                refused = true;
        }
        if (!refused) return;
        if (ran) throw std::runtime_error("flash_attention_varlen_backward: tensor-core path refused mid-call");
    }
    dispatch_fa_bwd(kFaModeVarlen, dt, Q.device,
                    Q.data, K.data, V.data, dO.data,
                    nullptr, cu_seqlens_q, cu_seqlens_k,
                    dQ.data, dK.data, dV.data,
                    total_q, total_k, D, D, D, D,
                    num_heads, head_dim, 0, causal, batch_size);
}

}  // namespace detail::cuda
}  // namespace brotensor
