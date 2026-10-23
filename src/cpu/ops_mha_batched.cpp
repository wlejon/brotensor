// ─── CPU multi-head attention, batched ops & masked mean-pool ──────────────
//
// Plain FP32 host loops decomposed from src/cpu/ops_impl.cpp.

#include <brotensor/tensor.h>
#include <brotensor/detail/cpu/thread_pool.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

void ensure_match(const ::brotensor::Tensor& src, ::brotensor::Tensor& dst) {
    if (dst.rows != src.rows || dst.cols != src.cols ||
        dst.dtype != ::brotensor::Dtype::FP32) {
        dst.resize(src.rows, src.cols, ::brotensor::Dtype::FP32);
    }
}

} // namespace

// ─── Multi-head self-attention ─────────────────────────────────────────────

void mha_forward(const ::brotensor::Tensor& X,
                 const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                 const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                 const ::brotensor::Tensor* bq, const ::brotensor::Tensor* bk,
                 const ::brotensor::Tensor* bv, const ::brotensor::Tensor* bo,
                 const float* d_mask, int num_heads,
                 ::brotensor::Tensor& Qh, ::brotensor::Tensor& Kh,
                 ::brotensor::Tensor& Vh, ::brotensor::Tensor& Attnh,
                 ::brotensor::Tensor& Yconcat, ::brotensor::Tensor& O) {
    using ::brotensor::Dtype;
    const int K = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    const int dh = (H > 0) ? D / H : 0;
    if (Qh.rows != H * K || Qh.cols != dh || Qh.dtype != Dtype::FP32) Qh.resize(H * K, dh, Dtype::FP32);
    if (Kh.rows != H * K || Kh.cols != dh || Kh.dtype != Dtype::FP32) Kh.resize(H * K, dh, Dtype::FP32);
    if (Vh.rows != H * K || Vh.cols != dh || Vh.dtype != Dtype::FP32) Vh.resize(H * K, dh, Dtype::FP32);
    if (Attnh.rows != H * K || Attnh.cols != K || Attnh.dtype != Dtype::FP32) Attnh.resize(H * K, K, Dtype::FP32);
    if (Yconcat.rows != K || Yconcat.cols != D || Yconcat.dtype != Dtype::FP32) Yconcat.resize(K, D, Dtype::FP32);
    if (O.rows != K || O.cols != D || O.dtype != Dtype::FP32) O.resize(K, D, Dtype::FP32);
    if (K == 0 || D == 0 || H == 0) return;

    const float* Xp  = X.host_f32();
    const float* Wqp = Wq.host_f32();
    const float* Wkp = Wk.host_f32();
    const float* Wvp = Wv.host_f32();
    const float* Wop = Wo.host_f32();
    float* Qp = Qh.host_f32_mut();
    float* Kp = Kh.host_f32_mut();
    float* Vp = Vh.host_f32_mut();
    float* Ap = Attnh.host_f32_mut();
    float* Yp = Yconcat.host_f32_mut();
    float* Op = O.host_f32_mut();

    const float* bqp = bq ? bq->host_f32() : nullptr;
    const float* bkp = bk ? bk->host_f32() : nullptr;
    const float* bvp = bv ? bv->host_f32() : nullptr;
    const float* bop = bo ? bo->host_f32() : nullptr;

    // Per-head Q/K/V projection: out(hh,i,j) = sum_k X(i,k) * W(hh*dh+j, k)
    // + (b ? b[hh*dh+j] : 0). Biases are flat length-D vectors indexed by
    // the full output column (hh*dh+j). Each head hh owns rows
    // [hh*K, hh*K+K) of Qh/Kh/Vh exclusively (X/W*/b* read-only), so the
    // head axis parallelizes with no cross-thread writes.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        const int row_off = hh * dh;
        for (int i = 0; i < K; ++i) {
            const float* xr = Xp + static_cast<std::size_t>(i) * D;
            const std::size_t out_row =
                (static_cast<std::size_t>(hh) * K + i) * dh;
            for (int j = 0; j < dh; ++j) {
                const int wrow = row_off + j;
                const float* wq = Wqp + static_cast<std::size_t>(wrow) * D;
                const float* wk = Wkp + static_cast<std::size_t>(wrow) * D;
                const float* wv = Wvp + static_cast<std::size_t>(wrow) * D;
                float aq = 0.0f, ak = 0.0f, av = 0.0f;
                for (int k = 0; k < D; ++k) {
                    aq += xr[k] * wq[k];
                    ak += xr[k] * wk[k];
                    av += xr[k] * wv[k];
                }
                if (bqp) aq += bqp[wrow];
                if (bkp) ak += bkp[wrow];
                if (bvp) av += bvp[wrow];
                Qp[out_row + j] = aq;
                Kp[out_row + j] = ak;
                Vp[out_row + j] = av;
            }
        }
    });

    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    // Per-head scores → masked row softmax → Attnh. Each hh owns rows
    // [hh*K, hh*K+K) of Attnh exclusively, so parallelizes over hh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < K; ++i) {
            float* arow = Ap + (static_cast<std::size_t>(hh) * K + i) * K;
            if (d_mask && d_mask[i] < 0.5f) {
                for (int j = 0; j < K; ++j) arow[j] = 0.0f;
                continue;
            }
            const float* qr = Qp + (static_cast<std::size_t>(hh) * K + i) * dh;
            float m = -1e30f;
            for (int j = 0; j < K; ++j) {
                if (d_mask && d_mask[j] < 0.5f) continue;
                const float* kr = Kp + (static_cast<std::size_t>(hh) * K + j) * dh;
                float s = 0.0f;
                for (int k = 0; k < dh; ++k) s += qr[k] * kr[k];
                s *= inv_sqrtdh;
                arow[j] = s;
                if (s > m) m = s;
            }
            float sum = 0.0f;
            for (int j = 0; j < K; ++j) {
                if (d_mask && d_mask[j] < 0.5f) { arow[j] = 0.0f; continue; }
                const float e = std::exp(arow[j] - m);
                arow[j] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int j = 0; j < K; ++j) arow[j] *= inv;
        }
    });

    // Per-head Attn @ V → Yconcat(i, hh*dh+k). Each hh writes only its own
    // column range [hh*dh, hh*dh+dh) of every row of Yconcat — different hh
    // never touch the same bytes, so this parallelizes over hh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < K; ++i) {
            const float* arow = Ap + (static_cast<std::size_t>(hh) * K + i) * K;
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < K; ++j) {
                    const float vv = Vp[(static_cast<std::size_t>(hh) * K + j) * dh + k];
                    acc += arow[j] * vv;
                }
                Yp[static_cast<std::size_t>(i) * D + (hh * dh + k)] = acc;
            }
        }
    });

    // Output projection O = Yconcat @ Wo^T + bo, zero invalid query rows.
    // Each i owns row i of O exclusively (Yconcat/Wo/bo read-only), so this
    // parallelizes over i.
    parallel_for(static_cast<std::size_t>(K), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        float* orow = Op + static_cast<std::size_t>(i) * D;
        if (d_mask && d_mask[i] < 0.5f) {
            for (int c = 0; c < D; ++c) orow[c] = 0.0f;
            return;
        }
        const float* yr = Yp + static_cast<std::size_t>(i) * D;
        for (int c = 0; c < D; ++c) {
            const float* wr = Wop + static_cast<std::size_t>(c) * D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += yr[k] * wr[k];
            if (bop) acc += bop[c];
            orow[c] = acc;
        }
    });
}

void mha_backward(const ::brotensor::Tensor& dO,
                  const ::brotensor::Tensor& X,
                  const ::brotensor::Tensor& Qh, const ::brotensor::Tensor& Kh,
                  const ::brotensor::Tensor& Vh, const ::brotensor::Tensor& Attnh,
                  const ::brotensor::Tensor& Yconcat,
                  const ::brotensor::Tensor& Wq, const ::brotensor::Tensor& Wk,
                  const ::brotensor::Tensor& Wv, const ::brotensor::Tensor& Wo,
                  const float* d_mask, int num_heads,
                  ::brotensor::Tensor& dX,
                  ::brotensor::Tensor& dWq, ::brotensor::Tensor& dWk,
                  ::brotensor::Tensor& dWv, ::brotensor::Tensor& dWo,
                  ::brotensor::Tensor* dbq, ::brotensor::Tensor* dbk,
                  ::brotensor::Tensor* dbv, ::brotensor::Tensor* dbo) {
    using ::brotensor::Dtype;
    const int K = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    const int dh = (H > 0) ? D / H : 0;
    if (dX.rows != K || dX.cols != D || dX.dtype != Dtype::FP32) {
        dX.resize(K, D, Dtype::FP32);
    }
    if (K == 0 || D == 0 || H == 0) return;

    const float inv_sqrtdh = 1.0f / std::sqrt(static_cast<float>(dh));

    const float* dOp = dO.host_f32();
    const float* Xp  = X.host_f32();
    const float* Qp  = Qh.host_f32();
    const float* Kp  = Kh.host_f32();
    const float* Vp  = Vh.host_f32();
    const float* Ap  = Attnh.host_f32();
    const float* Yp  = Yconcat.host_f32();
    const float* Wqp = Wq.host_f32();
    const float* Wkp = Wk.host_f32();
    const float* Wvp = Wv.host_f32();
    const float* Wop = Wo.host_f32();
    float* dXp  = dX.host_f32_mut();
    float* dWqp = dWq.host_f32_mut();
    float* dWkp = dWk.host_f32_mut();
    float* dWvp = dWv.host_f32_mut();
    float* dWop = dWo.host_f32_mut();

    std::vector<float> dYc(static_cast<std::size_t>(K) * D, 0.0f);
    std::vector<float> dAttn(static_cast<std::size_t>(H) * K * K, 0.0f);
    std::vector<float> dVh(static_cast<std::size_t>(H) * K * dh, 0.0f);
    std::vector<float> dScores(static_cast<std::size_t>(H) * K * K, 0.0f);
    std::vector<float> dQh(static_cast<std::size_t>(H) * K * dh, 0.0f);
    std::vector<float> dKh(static_cast<std::size_t>(H) * K * dh, 0.0f);

    // dYconcat (overwrite, zero on invalid rows). Each i owns row i of dYc
    // exclusively (Wo/dO read-only), so this parallelizes over i.
    parallel_for(static_cast<std::size_t>(K), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        const bool valid = !(d_mask && d_mask[i] < 0.5f);
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            if (valid) {
                for (int c = 0; c < D; ++c)
                    acc += Wop[static_cast<std::size_t>(c) * D + k] *
                           dOp[static_cast<std::size_t>(i) * D + c];
            }
            dYc[static_cast<std::size_t>(i) * D + k] = acc;
        }
    });
    // dWo (accumulate).
    for (int i = 0; i < K; ++i) {
        if (d_mask && d_mask[i] < 0.5f) continue;
        const float* dOi = dOp + static_cast<std::size_t>(i) * D;
        const float* Yi  = Yp  + static_cast<std::size_t>(i) * D;
        for (int c = 0; c < D; ++c) {
            const float dov = dOi[c];
            float* dWrow = dWop + static_cast<std::size_t>(c) * D;
            for (int k = 0; k < D; ++k) {
                dWrow[k] += dov * Yi[k];
            }
        }
    }

    // dbo[c] += sum over valid rows of dO[i,c] (caller zeros).
    if (dbo) {
        float* dbop = dbo->host_f32_mut();
        for (int i = 0; i < K; ++i) {
            if (d_mask && d_mask[i] < 0.5f) continue;
            const float* dOi = dOp + static_cast<std::size_t>(i) * D;
            for (int c = 0; c < D; ++c) {
                dbop[c] += dOi[c];
            }
        }
    }

    // Per-head dAttn and dVh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                float acc = 0.0f;
                for (int k = 0; k < dh; ++k) {
                    const float dy = dYc[static_cast<std::size_t>(i) * D + (hh * dh + k)];
                    const float vv = Vp[(static_cast<std::size_t>(hh) * K + j) * dh + k];
                    acc += dy * vv;
                }
                dAttn[(static_cast<std::size_t>(hh) * K + i) * K + j] = acc;
            }
        }
        for (int i = 0; i < K; ++i) {
            const float* Arow = Ap + (static_cast<std::size_t>(hh) * K + i) * K;
            const float* dYseg = dYc.data() + static_cast<std::size_t>(i) * D + hh * dh;
            for (int j = 0; j < K; ++j) {
                const float av = Arow[j];
                float* dVrow = dVh.data() + (static_cast<std::size_t>(hh) * K + j) * dh;
                for (int k = 0; k < dh; ++k) {
                    dVrow[k] += av * dYseg[k];
                }
            }
        }
    });

    // Per-head softmax backward → dScores.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < K; ++i) {
            const float* prow  = Ap + (static_cast<std::size_t>(hh) * K + i) * K;
            const float* dprow = dAttn.data() + (static_cast<std::size_t>(hh) * K + i) * K;
            float* drow = dScores.data() + (static_cast<std::size_t>(hh) * K + i) * K;
            if (d_mask && d_mask[i] < 0.5f) {
                for (int j = 0; j < K; ++j) drow[j] = 0.0f;
                continue;
            }
            float dot = 0.0f;
            for (int j = 0; j < K; ++j) dot += dprow[j] * prow[j];
            for (int j = 0; j < K; ++j) {
                if (d_mask && d_mask[j] < 0.5f) drow[j] = 0.0f;
                else drow[j] = prow[j] * (dprow[j] - dot) * inv_sqrtdh;
            }
        }
    });

    // Per-head dQh and dKh.
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < K; ++i) {
            for (int k = 0; k < dh; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < K; ++j) {
                    const float ds = dScores[(static_cast<std::size_t>(hh) * K + i) * K + j];
                    const float kk = Kp[(static_cast<std::size_t>(hh) * K + j) * dh + k];
                    acc += ds * kk;
                }
                dQh[(static_cast<std::size_t>(hh) * K + i) * dh + k] = acc;
            }
        }
        for (int i = 0; i < K; ++i) {
            const float* dSrow = dScores.data() + (static_cast<std::size_t>(hh) * K + i) * K;
            const float* Qseg = Qp + (static_cast<std::size_t>(hh) * K + i) * dh;
            for (int j = 0; j < K; ++j) {
                const float ds = dSrow[j];
                float* dKrow = dKh.data() + (static_cast<std::size_t>(hh) * K + j) * dh;
                for (int k = 0; k < dh; ++k) {
                    dKrow[k] += ds * Qseg[k];
                }
            }
        }
    });

    // dbq/dbk/dbv accumulate: db[hh*dh+j] += sum_i dQh/dKh/dVh(hh,i,j).
    if (dbq || dbk || dbv) {
        float* dbqp = dbq ? dbq->host_f32_mut() : nullptr;
        float* dbkp = dbk ? dbk->host_f32_mut() : nullptr;
        float* dbvp = dbv ? dbv->host_f32_mut() : nullptr;
        parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
            const int hh = static_cast<int>(hhi);
            float* dbq_row = dbqp ? dbqp + hh * dh : nullptr;
            float* dbk_row = dbkp ? dbkp + hh * dh : nullptr;
            float* dbv_row = dbvp ? dbvp + hh * dh : nullptr;
            for (int i = 0; i < K; ++i) {
                const std::size_t base = (static_cast<std::size_t>(hh) * K + i) * dh;
                for (int j = 0; j < dh; ++j) {
                    if (dbq_row) dbq_row[j] += dQh[base + j];
                    if (dbk_row) dbk_row[j] += dKh[base + j];
                    if (dbv_row) dbv_row[j] += dVh[base + j];
                }
            }
        });
    }

    // dWq/dWk/dWv accumulate: dW(hh*dh+j, k) += sum_i dQh(hh,i,j) * X(i,k).
    parallel_for(static_cast<std::size_t>(H), [&](std::size_t hhi) {
        const int hh = static_cast<int>(hhi);
        for (int i = 0; i < K; ++i) {
            const float* Xi = Xp + static_cast<std::size_t>(i) * D;
            const std::size_t base = (static_cast<std::size_t>(hh) * K + i) * dh;
            for (int j = 0; j < dh; ++j) {
                const int wrow = hh * dh + j;
                const float qv = dQh[base + j];
                const float kv = dKh[base + j];
                const float vv = dVh[base + j];
                float* dWqrow = dWqp + static_cast<std::size_t>(wrow) * D;
                float* dWkrow = dWkp + static_cast<std::size_t>(wrow) * D;
                float* dWvrow = dWvp + static_cast<std::size_t>(wrow) * D;
                for (int k = 0; k < D; ++k) {
                    const float xv = Xi[k];
                    dWqrow[k] += qv * xv;
                    dWkrow[k] += kv * xv;
                    dWvrow[k] += vv * xv;
                }
            }
        }
    });

    // dX(i,k) = sum over heads, j of dQh*Wq + dKh*Wk + dVh*Wv at (hh*dh+j, k).
    parallel_for(static_cast<std::size_t>(K), [&](std::size_t ii) {
        const int i = static_cast<int>(ii);
        for (int k = 0; k < D; ++k) {
            float acc = 0.0f;
            for (int hh = 0; hh < H; ++hh) {
                for (int j = 0; j < dh; ++j) {
                    const int wrow = hh * dh + j;
                    const std::size_t widx = static_cast<std::size_t>(wrow) * D + k;
                    const float gq = dQh[(static_cast<std::size_t>(hh) * K + i) * dh + j];
                    const float gk = dKh[(static_cast<std::size_t>(hh) * K + i) * dh + j];
                    const float gv = dVh[(static_cast<std::size_t>(hh) * K + i) * dh + j];
                    acc += gq * Wqp[widx] + gk * Wkp[widx] + gv * Wvp[widx];
                }
            }
            dXp[static_cast<std::size_t>(i) * D + k] = acc;
        }
    });
}

// ─── Concat / split ────────────────────────────────────────────────────────

void concat_rows(const std::vector<const ::brotensor::Tensor*>& parts,
                 ::brotensor::Tensor& out) {
    int total = 0;
    for (const auto* p : parts) {
        if (p) total += p->size();
    }
    if (out.rows != total || out.cols != 1 ||
        out.dtype != ::brotensor::Dtype::FP32) {
        out.resize(total, 1, ::brotensor::Dtype::FP32);
    }
    if (total == 0) return;
    float* op = out.host_f32_mut();
    int off = 0;
    for (const auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        const float* pp = p->host_f32();
        for (int i = 0; i < n; ++i) op[off + i] = pp[i];
        off += n;
    }
}

void split_rows(const ::brotensor::Tensor& in,
                const std::vector<::brotensor::Tensor*>& parts) {
    const float* ip = in.host_f32();
    int off = 0;
    for (auto* p : parts) {
        if (!p) continue;
        const int n = p->size();
        if (n == 0) continue;
        float* pp = p->host_f32_mut();
        for (int i = 0; i < n; ++i) pp[i] = ip[off + i];
        off += n;
    }
}

// ─── Batched elementwise / linear ──────────────────────────────────────────

void add_inplace_batched(::brotensor::Tensor& Y_BD,
                         const ::brotensor::Tensor& X_BD) {
    const int n = Y_BD.size();
    assert(X_BD.size() == n);
    float* yp = Y_BD.host_f32_mut();
    const float* xp = X_BD.host_f32();
    for (int i = 0; i < n; ++i) yp[i] += xp[i];
}

void linear_forward_batched(const ::brotensor::Tensor& W,
                            const ::brotensor::Tensor& bias,
                            const ::brotensor::Tensor& X_BD,
                            ::brotensor::Tensor& Y_BD) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (Y_BD.rows != B || Y_BD.cols != out_dim ||
        Y_BD.dtype != ::brotensor::Dtype::FP32) {
        Y_BD.resize(B, out_dim, ::brotensor::Dtype::FP32);
    }
    if (B == 0 || out_dim == 0) return;

    const float* bp = bias.host_f32();
    const float* Xp = X_BD.host_f32();
    float* Yp = Y_BD.host_f32_mut();

    // 16-bit weights (FP32 activations): widen per element.
    if (W.dtype == ::brotensor::Dtype::FP16 ||
        W.dtype == ::brotensor::Dtype::BF16) {
        const bool bf = (W.dtype == ::brotensor::Dtype::BF16);
        const std::uint16_t* Wp = static_cast<const std::uint16_t*>(W.data);
        for (int b = 0; b < B; ++b) {
            const float* xr = Xp + static_cast<std::size_t>(b) * in_dim;
            float* yr = Yp + static_cast<std::size_t>(b) * out_dim;
            for (int i = 0; i < out_dim; ++i) {
                const std::uint16_t* wr = Wp + static_cast<std::size_t>(i) * in_dim;
                float acc = bp[i];
                for (int j = 0; j < in_dim; ++j) {
                    const float w = bf ? ::brotensor::bf16_bits_to_fp32(wr[j])
                                       : ::brotensor::fp16_bits_to_fp32(wr[j]);
                    acc += w * xr[j];
                }
                yr[i] = acc;
            }
        }
        return;
    }

    const float* Wp = W.host_f32();
    for (int b = 0; b < B; ++b) {
        const float* xr = Xp + static_cast<std::size_t>(b) * in_dim;
        float* yr = Yp + static_cast<std::size_t>(b) * out_dim;
        for (int i = 0; i < out_dim; ++i) {
            const float* wr = Wp + static_cast<std::size_t>(i) * in_dim;
            float acc = bp[i];
            for (int j = 0; j < in_dim; ++j) acc += wr[j] * xr[j];
            yr[i] = acc;
        }
    }
}

void linear_backward_batched(const ::brotensor::Tensor& W,
                             const ::brotensor::Tensor& X_BD,
                             const ::brotensor::Tensor& dY_BD,
                             ::brotensor::Tensor& dX_BD,
                             ::brotensor::Tensor& dW,
                             ::brotensor::Tensor& dB) {
    const int out_dim = W.rows;
    const int in_dim  = W.cols;
    const int B       = X_BD.rows;
    if (dX_BD.rows != B || dX_BD.cols != in_dim ||
        dX_BD.dtype != ::brotensor::Dtype::FP32) {
        dX_BD.resize(B, in_dim, ::brotensor::Dtype::FP32);
    }
    if (B == 0) return;

    const float* Wp  = W.host_f32();
    const float* Xp  = X_BD.host_f32();
    const float* dYp = dY_BD.host_f32();
    float* dXp = dX_BD.host_f32_mut();
    float* dWp = dW.host_f32_mut();
    float* dBp = dB.host_f32_mut();

    // dX[b] = W^T * dY[b]   (overwrite).
    for (int b = 0; b < B; ++b) {
        const float* dyr = dYp + static_cast<std::size_t>(b) * out_dim;
        float* dxr = dXp + static_cast<std::size_t>(b) * in_dim;
        for (int j = 0; j < in_dim; ++j) dxr[j] = 0.0f;
        for (int i = 0; i < out_dim; ++i) {
            const float g = dyr[i];
            const float* wr = Wp + static_cast<std::size_t>(i) * in_dim;
            for (int j = 0; j < in_dim; ++j) dxr[j] += wr[j] * g;
        }
    }
    // dW += sum_b dY[b] * X[b]^T;  dB += sum_b dY[b]   (accumulate).
    for (int b = 0; b < B; ++b) {
        const float* dyr = dYp + static_cast<std::size_t>(b) * out_dim;
        const float* xr  = Xp  + static_cast<std::size_t>(b) * in_dim;
        for (int i = 0; i < out_dim; ++i) {
            const float g = dyr[i];
            float* dwr = dWp + static_cast<std::size_t>(i) * in_dim;
            for (int j = 0; j < in_dim; ++j) dwr[j] += g * xr[j];
            dBp[i] += g;
        }
    }
}

void relu_forward_batched(const ::brotensor::Tensor& X_BD,
                          ::brotensor::Tensor& Y_BD) {
    ensure_match(X_BD, Y_BD);
    const int n = X_BD.size();
    const float* xp = X_BD.host_f32();
    float* yp = Y_BD.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
}

void relu_backward_batched(const ::brotensor::Tensor& X_BD,
                           const ::brotensor::Tensor& dY_BD,
                           ::brotensor::Tensor& dX_BD) {
    ensure_match(X_BD, dX_BD);
    const int n = X_BD.size();
    const float* xp  = X_BD.host_f32();
    const float* dYp = dY_BD.host_f32();
    float* dXp = dX_BD.host_f32_mut();
    for (int i = 0; i < n; ++i) dXp[i] = xp[i] > 0.0f ? dYp[i] : 0.0f;
}

void tanh_forward_batched(const ::brotensor::Tensor& X_BD,
                          ::brotensor::Tensor& Y_BD) {
    ensure_match(X_BD, Y_BD);
    const int n = X_BD.size();
    const float* xp = X_BD.host_f32();
    float* yp = Y_BD.host_f32_mut();
    for (int i = 0; i < n; ++i) yp[i] = std::tanh(xp[i]);
}

void tanh_backward_batched(const ::brotensor::Tensor& Y_BD,
                           const ::brotensor::Tensor& dY_BD,
                           ::brotensor::Tensor& dX_BD) {
    ensure_match(Y_BD, dX_BD);
    const int n = Y_BD.size();
    const float* yp  = Y_BD.host_f32();
    const float* dYp = dY_BD.host_f32();
    float* dXp = dX_BD.host_f32_mut();
    for (int i = 0; i < n; ++i) {
        const float yc = std::clamp(yp[i], -1.0f, 1.0f);
        dXp[i] = dYp[i] * (1.0f - yc * yc);
    }
}

// ─── Batched per-sample losses ─────────────────────────────────────────────

void mse_vec_per_sample(const ::brotensor::Tensor& pred,
                        const ::brotensor::Tensor& target,
                        ::brotensor::Tensor& dPred,
                        ::brotensor::Tensor& loss_per_sample) {
    const int B = pred.size();
    if (dPred.rows != pred.rows || dPred.cols != pred.cols ||
        dPred.dtype != ::brotensor::Dtype::FP32) {
        dPred.resize(pred.rows, pred.cols, ::brotensor::Dtype::FP32);
    }
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != ::brotensor::Dtype::FP32) {
        loss_per_sample.resize(B, 1, ::brotensor::Dtype::FP32);
    }
    if (B == 0) return;
    const float* pp = pred.host_f32();
    const float* tp = target.host_f32();
    float* dp = dPred.host_f32_mut();
    float* lp = loss_per_sample.host_f32_mut();
    for (int b = 0; b < B; ++b) {
        const float d = pp[b] - tp[b];
        dp[b] = d;
        lp[b] = 0.5f * d * d;
    }
}

void softmax_xent_fused_batched(const ::brotensor::Tensor& logits_BL,
                                const ::brotensor::Tensor& target_BL,
                                const float* d_mask_BL,
                                const int* d_head_offsets,
                                int n_heads,
                                ::brotensor::Tensor& probs_BL,
                                ::brotensor::Tensor& dLogits_BL,
                                ::brotensor::Tensor& loss_per_sample) {
    const int B     = logits_BL.rows;
    const int n_act = logits_BL.cols;
    if (probs_BL.rows != B || probs_BL.cols != n_act ||
        probs_BL.dtype != ::brotensor::Dtype::FP32) {
        probs_BL.resize(B, n_act, ::brotensor::Dtype::FP32);
    }
    if (dLogits_BL.rows != B || dLogits_BL.cols != n_act ||
        dLogits_BL.dtype != ::brotensor::Dtype::FP32) {
        dLogits_BL.resize(B, n_act, ::brotensor::Dtype::FP32);
    }
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != ::brotensor::Dtype::FP32) {
        loss_per_sample.resize(B, 1, ::brotensor::Dtype::FP32);
    }
    if (B == 0 || n_act == 0 || n_heads <= 0) return;

    const float* lp = logits_BL.host_f32();
    const float* tp = target_BL.host_f32();
    float* pp = probs_BL.host_f32_mut();
    float* dp = dLogits_BL.host_f32_mut();
    float* lossp = loss_per_sample.host_f32_mut();

    for (int b = 0; b < B; ++b) {
        // Overwrite loss before per-head accumulation.
        float sample_loss = 0.0f;
        const int row_off = b * n_act;
        for (int h = 0; h < n_heads; ++h) {
            const int off = d_head_offsets[h];
            const int end = d_head_offsets[h + 1];
            const int len = end - off;
            const float* lr = lp + row_off + off;
            const float* tr = tp + row_off + off;
            const float* mr = d_mask_BL ? (d_mask_BL + row_off + off) : nullptr;
            float* pr = pp + row_off + off;
            float* dr = dp + row_off + off;

            float m = -1e30f;
            for (int i = 0; i < len; ++i) {
                if (mr && mr[i] < 0.5f) continue;
                if (lr[i] > m) m = lr[i];
            }
            float sum = 0.0f;
            for (int i = 0; i < len; ++i) {
                if (mr && mr[i] < 0.5f) { pr[i] = 0.0f; continue; }
                const float e = std::exp(lr[i] - m);
                pr[i] = e;
                sum += e;
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int i = 0; i < len; ++i) {
                if (mr && mr[i] < 0.5f) { dr[i] = 0.0f; continue; }
                const float p = pr[i] * inv;
                pr[i] = p;
                const float t = tr[i];
                if (t > 0.0f) {
                    const float pc = p > 1e-12f ? p : 1e-12f;
                    sample_loss -= t * std::log(pc);
                }
                dr[i] = p - t;
            }
        }
        lossp[b] = sample_loss;
    }
}

void bce_with_logits_fused_batched(const ::brotensor::Tensor& logits_BL,
                                   const ::brotensor::Tensor& target_BL,
                                   const float* d_mask_BL,
                                   float pos_weight,
                                   ::brotensor::Tensor& probs_BL,
                                   ::brotensor::Tensor& dLogits_BL,
                                   ::brotensor::Tensor& loss_per_sample) {
    const int B = logits_BL.rows;
    const int L = logits_BL.cols;
    if (probs_BL.rows != B || probs_BL.cols != L ||
        probs_BL.dtype != ::brotensor::Dtype::FP32) {
        probs_BL.resize(B, L, ::brotensor::Dtype::FP32);
    }
    if (dLogits_BL.rows != B || dLogits_BL.cols != L ||
        dLogits_BL.dtype != ::brotensor::Dtype::FP32) {
        dLogits_BL.resize(B, L, ::brotensor::Dtype::FP32);
    }
    if (loss_per_sample.rows != B || loss_per_sample.cols != 1 ||
        loss_per_sample.dtype != ::brotensor::Dtype::FP32) {
        loss_per_sample.resize(B, 1, ::brotensor::Dtype::FP32);
    }
    if (B == 0 || L == 0) return;

    const float* lp = logits_BL.host_f32();
    const float* tp = target_BL.host_f32();
    float* pp = probs_BL.host_f32_mut();
    float* dp = dLogits_BL.host_f32_mut();
    float* lossp = loss_per_sample.host_f32_mut();
    const float w = pos_weight;

    for (int b = 0; b < B; ++b) {
        float sample_loss = 0.0f;
        const int row_off = b * L;
        const float* lr = lp + row_off;
        const float* tr = tp + row_off;
        const float* mr = d_mask_BL ? (d_mask_BL + row_off) : nullptr;
        float* pr = pp + row_off;
        float* dr = dp + row_off;
        for (int i = 0; i < L; ++i) {
            if (mr && mr[i] < 0.5f) {
                pr[i] = 0.0f;
                dr[i] = 0.0f;
                continue;
            }
            const float z = lr[i];
            const float y = tr[i];
            // softplus(x) = max(x, 0) + log1p(exp(-|x|))
            const float az = std::fabs(z);
            const float sp_neg = (z > 0.0f ? 0.0f : -z) + std::log1p(std::exp(-az));
            const float sp_pos = (z > 0.0f ? z      : 0.0f) + std::log1p(std::exp(-az));
            sample_loss += w * y * sp_neg + (1.0f - y) * sp_pos;
            // s = sigmoid(z), numerically stable.
            const float s = z >= 0.0f
                ? 1.0f / (1.0f + std::exp(-z))
                : std::exp(z) / (1.0f + std::exp(z));
            pr[i] = s;
            dr[i] = s * (w * y + 1.0f - y) - w * y;
        }
        lossp[b] = sample_loss;
    }
}

// ─── Masked mean-pool ──────────────────────────────────────────────────────

void build_slot_mask(const ::brotensor::Tensor& x, int offset, int K, int stride,
                     ::brotensor::Tensor& mask) {
    if (mask.rows != K || mask.cols != 1 ||
        mask.dtype != ::brotensor::Dtype::FP32) {
        mask.resize(K, 1, ::brotensor::Dtype::FP32);
    }
    const float* xp = x.host_f32();
    float* mp = mask.host_f32_mut();
    for (int k = 0; k < K; ++k) {
        mp[k] = (xp[offset + static_cast<std::size_t>(k) * stride] > 0.5f)
                    ? 1.0f : 0.0f;
    }
}

void copy_d2d(const ::brotensor::Tensor& src, int src_off,
              ::brotensor::Tensor& dst, int dst_off, int n) {
    const float* sp = src.host_f32();
    float* dp = dst.host_f32_mut();
    std::memcpy(dp + dst_off, sp + src_off, static_cast<std::size_t>(n) * sizeof(float));
}

void copy_d2d_strided(const ::brotensor::Tensor& src, int src_off, int src_pitch,
                      ::brotensor::Tensor& dst, int dst_off, int dst_pitch,
                      int width, int height) {
    const float* sp = src.host_f32() + src_off;
    float* dp = dst.host_f32_mut() + dst_off;
    for (int r = 0; r < height; ++r) {
        std::memcpy(dp + static_cast<std::size_t>(r) * dst_pitch,
                    sp + static_cast<std::size_t>(r) * src_pitch,
                    static_cast<std::size_t>(width) * sizeof(float));
    }
}

void masked_mean_pool_forward(const ::brotensor::Tensor& X, const float* d_mask,
                              ::brotensor::Tensor& y) {
    const int K = X.rows;
    const int D = X.cols;
    if (y.rows != D || y.cols != 1 || y.dtype != ::brotensor::Dtype::FP32) {
        y.resize(D, 1, ::brotensor::Dtype::FP32);
    }
    const float* Xp = X.host_f32();
    float* yp = y.host_f32_mut();
    for (int j = 0; j < D; ++j) yp[j] = 0.0f;
    int num_valid = 0;
    for (int k = 0; k < K; ++k) {
        if (d_mask && d_mask[k] < 0.5f) continue;
        const float* row = Xp + static_cast<std::size_t>(k) * D;
        for (int j = 0; j < D; ++j) yp[j] += row[j];
        ++num_valid;
    }
    if (num_valid > 0) {
        const float inv = 1.0f / static_cast<float>(num_valid);
        for (int j = 0; j < D; ++j) yp[j] *= inv;
    }
}

void masked_mean_pool_backward(const ::brotensor::Tensor& dY, const float* d_mask,
                               int K, ::brotensor::Tensor& dX) {
    const int D = dY.size();
    if (dX.rows != K || dX.cols != D || dX.dtype != ::brotensor::Dtype::FP32) {
        dX.resize(K, D, ::brotensor::Dtype::FP32);
    }
    const float* dYp = dY.host_f32();
    float* dXp = dX.host_f32_mut();
    int num_valid = 0;
    for (int k = 0; k < K; ++k) {
        if (!d_mask || d_mask[k] >= 0.5f) ++num_valid;
    }
    const float inv = num_valid > 0 ? 1.0f / static_cast<float>(num_valid) : 0.0f;
    for (int k = 0; k < K; ++k) {
        float* row = dXp + static_cast<std::size_t>(k) * D;
        const bool valid = (!d_mask || d_mask[k] >= 0.5f) && num_valid > 0;
        for (int j = 0; j < D; ++j) row[j] = valid ? dYp[j] * inv : 0.0f;
    }
}

} // namespace brotensor::detail::cpu
