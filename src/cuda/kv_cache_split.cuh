#pragma once

// ---------------------------------------------------------------------------
// Split-K decode attention (L_q == 1, head_dim % 32 == 0, head_dim <= 256).
//
// Included inside the anonymous namespace of src/cuda/kv_cache.cu.
//
// The single-block-per-head kernels leave a 128-SM GPU almost idle at
// decode (16 q heads = 16 blocks) and crawl each K row with one thread, so
// they run at ~2% of HBM bandwidth. This pair fixes both:
//
//   stage 1  grid (n_splits, n_q_heads) — each block runs the online-softmax
//            core over its slice of the key range and writes an unnormalized
//            partial record [m, sum, acc[head_dim]] to a workspace. Keys are
//            scored a warp at a time: the 32 lanes split head_dim, so a K row
//            is one coalesced 2*head_dim-byte read instead of a serial crawl.
//   stage 2  grid (n_q_heads) — rescales the split partials to the global max
//            and combines them (the standard flash-decoding merge).
//
// One template serves both the length-truncated and the masked op; validity
// is `kg < valid_len` or `mask[kg] > 0.5f` per key. The arithmetic sequence
// is otherwise identical, and split boundaries derive from the cache
// CAPACITY (not valid_len) in both, so masked-with-prefix-mask stays
// bit-identical to truncated — the contract the graph decode path relies on.
// Masked-out keys score -1e30 (softmax weight exactly 0.0f) and their K/V
// rows are never read, so garbage (even NaN) in dead rows cannot propagate.
// ---------------------------------------------------------------------------

constexpr int FAD_SPLIT_BLOCK  = 128;
constexpr int FAD_MAX_SPLITS   = 16;
constexpr int FAD_SPLIT_MAX_HD = 256;

template <typename T, bool MASKED, bool DEVLEN = false>
__global__ void fad_split_partials_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        const float* __restrict__ mask,   // MASKED only, else unused
        float* __restrict__ ws,           // (n_q_heads*n_splits) x (head_dim+2)
        int cap, int valid_len,           // valid_len used when !MASKED
        int Dq, int Dkv, int head_dim, int group,
        int tiles_per_split, int n_splits,
        float softcap, int window,
        const int* __restrict__ d_len = nullptr) {   // DEVLEN: valid length on the device
    __shared__ float scores[FAD_KTILE];
    __shared__ float bcast[2];

    if constexpr (DEVLEN) {
        valid_len = *d_len;
        if (valid_len > cap) valid_len = cap;
        const int n_tiles = (valid_len + FAD_KTILE - 1) / FAD_KTILE;
        tiles_per_split = (n_tiles + n_splits - 1) / n_splits;
        if (tiles_per_split < 1) tiles_per_split = 1;
    }

    const int h    = blockIdx.y;
    const int spl  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr int N_WARPS = FAD_SPLIT_BLOCK / 32;
    constexpr int KPW     = FAD_KTILE / N_WARPS;   // keys per warp per tile

    const int q_head_off  = h * head_dim;
    const int kv_head_off = (h / group) * head_dim;
    const float inv_sqrt  = rsqrtf(static_cast<float>(head_dim));
    const int e = head_dim / 32;                   // dot elems per lane

    int lo = 0;
    if (window > 0) {
        int p_max = MASKED ? -1 : (valid_len - 1);
        if (MASKED) {
            float lpm = -1.0f;
            for (int kg = tid; kg < cap; kg += FAD_SPLIT_BLOCK) {
                if (mask[kg] > 0.5f) lpm = fmaxf(lpm, static_cast<float>(kg));
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lpm = fmaxf(lpm, __shfl_down_sync(0xffffffffu, lpm, off));
            }
            if (lane == 0) scores[warp] = lpm;
            __syncthreads();
            if (warp == 0) {
                float m = (lane < N_WARPS) ? scores[lane] : -1.0f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
                }
                if (lane == 0) bcast[0] = m;
            }
            __syncthreads();
            p_max = static_cast<int>(bcast[0]);
            __syncthreads();
        }
        if (p_max >= 0) { lo = p_max - window + 1; if (lo < 0) lo = 0; }
    }

    float qreg[FAD_SPLIT_MAX_HD / 32];
    #pragma unroll
    for (int i = 0; i < FAD_SPLIT_MAX_HD / 32; ++i) qreg[i] = 0.0f;
    for (int i = 0; i < e; ++i) {
        qreg[i] = fadm_ld(Q[q_head_off + lane * e + i]);
    }

    constexpr int MAX_SLOTS = FAD_SPLIT_MAX_HD / FAD_SPLIT_BLOCK;
    float partial[MAX_SLOTS];
    #pragma unroll
    for (int i = 0; i < MAX_SLOTS; ++i) partial[i] = 0.0f;
    float run_max = -1e30f;
    float run_sum = 0.0f;

    const int k_begin = spl * tiles_per_split * FAD_KTILE;
    int k_end = k_begin + tiles_per_split * FAD_KTILE;
    if (k_end > cap) k_end = cap;
    if (DEVLEN && k_end > valid_len) k_end = valid_len;

    for (int k0 = k_begin; k0 < k_end; k0 += FAD_KTILE) {
        if (!MASKED && k0 >= valid_len) break;
        const int klen = (k_end - k0) < FAD_KTILE ? (k_end - k0) : FAD_KTILE;

        for (int j = 0; j < KPW; ++j) {
            const int t = warp * KPW + j;
            if (t >= klen) break;
            const int kg = k0 + t;
            const bool valid = (kg >= lo) &&
                               (MASKED ? (mask[kg] > 0.5f) : (kg < valid_len));
            if (!valid) {
                if (lane == 0) scores[t] = -1e30f;
                continue;
            }
            const T* krow = K + static_cast<size_t>(kg) * Dkv + kv_head_off +
                            lane * e;
            float dot = 0.0f;
            if ((e & 1) == 0) {
                using T2 = typename FadmT2<T>::type;
                #pragma unroll
                for (int i = 0; i < FAD_SPLIT_MAX_HD / 32; i += 2) {
                    if (i >= e) break;
                    const T2 kv2 = *reinterpret_cast<const T2*>(krow + i);
                    dot += qreg[i] * fadm_ld(kv2.x) +
                           qreg[i + 1] * fadm_ld(kv2.y);
                }
            } else {
                for (int i = 0; i < e; ++i) dot += qreg[i] * fadm_ld(krow[i]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                dot += __shfl_down_sync(0xffffffffu, dot, off);
            }
            if (lane == 0) scores[t] = fad_softcap(dot * inv_sqrt, softcap);
        }
        __syncthreads();

        if (warp == 0) {   // tile max
            float m = -1e30f;
            for (int t = lane; t < klen; t += 32) m = fmaxf(m, scores[t]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
            }
            if (lane == 0) bcast[0] = m;
        }
        __syncthreads();
        const float m_new = fmaxf(run_max, bcast[0]);
        const bool tile_empty = (m_new <= -1e29f);

        if (tid < klen) {
            scores[tid] = tile_empty ? 0.0f : __expf(scores[tid] - m_new);
        }
        __syncthreads();
        if (warp == 0) {   // tile weight sum
            float s = 0.0f;
            for (int t = lane; t < klen; t += 32) s += scores[t];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                s += __shfl_down_sync(0xffffffffu, s, off);
            }
            if (lane == 0) bcast[1] = s;
        }
        __syncthreads();
        const float tile_sum = bcast[1];
        const float alpha = (run_max <= -1e29f) ? 0.0f
                                                : __expf(run_max - m_new);

        int slot = 0;
        for (int d = tid; d < head_dim; d += FAD_SPLIT_BLOCK, ++slot) {
            float acc = alpha * partial[slot];
            #pragma unroll 4
            for (int t = 0; t < klen; ++t) {
                const float s = scores[t];
                if (s == 0.0f) continue;
                acc += s * fadm_ld(V[static_cast<size_t>(k0 + t) * Dkv +
                                     kv_head_off + d]);
            }
            partial[slot] = acc;
        }
        run_max = m_new;
        run_sum = alpha * run_sum + tile_sum;
        __syncthreads();
    }

    float* rec = ws + (static_cast<size_t>(h) * n_splits + spl) *
                          (head_dim + 2);
    if (tid == 0) {
        rec[0] = run_max;
        rec[1] = run_sum;
    }
    int slot = 0;
    for (int d = tid; d < head_dim; d += FAD_SPLIT_BLOCK, ++slot) {
        rec[2 + d] = partial[slot];
    }
}

template <typename T>
__global__ void fad_split_reduce_kernel(const float* __restrict__ ws,
                                        T* __restrict__ Out,
                                        int head_dim, int n_splits) {
    const int h       = blockIdx.x;
    const int tid     = threadIdx.x;
    const int rec_len = head_dim + 2;
    const float* base = ws + static_cast<size_t>(h) * n_splits * rec_len;

    float m_g = -1e30f;
    for (int s = 0; s < n_splits; ++s) m_g = fmaxf(m_g, base[s * rec_len]);

    float f[FAD_MAX_SPLITS];
    float den = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float m_s = base[s * rec_len];
        const float w   = (m_s <= -1e29f) ? 0.0f : __expf(m_s - m_g);
        f[s] = w;
        den += w * base[s * rec_len + 1];
    }
    const float inv = (den > 0.0f) ? (1.0f / den) : 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float num = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            if (f[s] == 0.0f) continue;   // empty split: acc rows are stale
            num += f[s] * base[s * rec_len + 2 + d];
        }
        fadm_st(Out[h * head_dim + d], num * inv);
    }
}

inline float* fad_split_workspace(size_t need_floats) {
    constexpr int MAX_DEVICES = 16;
    static float* buf[MAX_DEVICES]   = {};
    static size_t cap_f[MAX_DEVICES] = {};
    int dev = 0;
    BROTENSOR_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev < 0 || dev >= MAX_DEVICES) {
        throw std::runtime_error(
            "flash_attention_decode: device ordinal out of range");
    }
    if (cap_f[dev] < need_floats) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        BROTENSOR_CUDA_CHECK(cudaStreamIsCapturing(cur_stream(), &st));
        if (st != cudaStreamCaptureStatusNone) {
            throw std::runtime_error(
                "flash_attention_decode(_masked): split-K workspace cannot "
                "grow during CUDA graph capture — run the op once eagerly "
                "(warm-up) before capturing");
        }
        const size_t want = need_floats > 2 * cap_f[dev] ? need_floats : 2 * cap_f[dev];
        buf[dev]   = nullptr;
        cap_f[dev] = 0;
        BROTENSOR_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf[dev]),
                                        want * sizeof(float)));
        cap_f[dev] = want;
    }
    return buf[dev];
}

inline void fad_split_dims(int cap, int& n_splits, int& tiles_per_split) {
    const int n_tiles = (cap + FAD_KTILE - 1) / FAD_KTILE;
    n_splits = n_tiles < FAD_MAX_SPLITS ? n_tiles : FAD_MAX_SPLITS;
    tiles_per_split = (n_tiles + n_splits - 1) / n_splits;
}

inline bool fad_split_eligible(int head_dim) {
    return head_dim % 32 == 0 && head_dim <= FAD_SPLIT_MAX_HD;
}
