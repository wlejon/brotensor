#include "cpu_jit.h"
#include <brotensor/detail/cpu/thread_pool.h>
#include <brotensor/tensor.h>
#include <brotensor/ops.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace brotensor::detail::cpu::jit {
namespace {

static inline void ref_dequant_q8_0_block(const void* blk_ptr, float* out32) {
    uint16_t d_raw;
    std::memcpy(&d_raw, blk_ptr, 2);
    const float d = fp16_bits_to_fp32(d_raw);
    const int8_t* qs = reinterpret_cast<const int8_t*>(static_cast<const char*>(blk_ptr) + 2);
    for (int i = 0; i < 32; ++i) {
        out32[i] = static_cast<float>(qs[i]) * d;
    }
}

static inline void ref_dequant_q4k_block(const void* blk_ptr, float* out256) {
    const uint8_t* p = static_cast<const uint8_t*>(blk_ptr);
    uint16_t d_raw, dmin_raw;
    std::memcpy(&d_raw, p, 2);
    std::memcpy(&dmin_raw, p + 2, 2);
    const float d = fp16_bits_to_fp32(d_raw);
    const float dmin = fp16_bits_to_fp32(dmin_raw);

    const uint8_t* scales = p + 4;
    const uint8_t* qs = p + 16;

    uint8_t sc[8], m[8];
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = scales[j]     & 0x3F;
            m[j]  = scales[j + 4] & 0x3F;
        } else {
            sc[j] = static_cast<uint8_t>((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
            m[j]  = static_cast<uint8_t>((scales[j + 4] >> 4)   | ((scales[j - 0] >> 6) << 4));
        }
    }

    for (int p_idx = 0; p_idx < 4; ++p_idx) {
        const int is_lo = 2 * p_idx;
        const int is_hi = 2 * p_idx + 1;
        const float w_lo = static_cast<float>(sc[is_lo]) * d;
        const float w_hi = static_cast<float>(sc[is_hi]) * d;
        const float b_lo = static_cast<float>(m [is_lo]) * dmin;
        const float b_hi = static_cast<float>(m [is_hi]) * dmin;

        for (int l = 0; l < 32; ++l) {
            const uint8_t byte = qs[p_idx * 32 + l];
            const int n_lo = byte & 0x0F;
            const int n_hi = (byte >> 4) & 0x0F;
            out256[is_lo * 32 + l] = w_lo * static_cast<float>(n_lo) - b_lo;
            out256[is_hi * 32 + l] = w_hi * static_cast<float>(n_hi) - b_hi;
        }
    }
}

static void ref_gemv_q8_0(const void* W, const float* X, float* Y, int N, int K) {
    const int bpr = K / 32;
    const char* W_bytes = static_cast<const char*>(W);
    float blk_buf[32];
    for (int r = 0; r < N; ++r) {
        double acc = 0.0;
        const char* row_ptr = W_bytes + static_cast<size_t>(r) * bpr * 34;
        for (int b = 0; b < bpr; ++b) {
            ref_dequant_q8_0_block(row_ptr + b * 34, blk_buf);
            const float* x_sub = X + b * 32;
            for (int i = 0; i < 32; ++i) {
                acc += static_cast<double>(blk_buf[i]) * static_cast<double>(x_sub[i]);
            }
        }
        Y[r] = static_cast<float>(acc);
    }
}

static void ref_gemv_q4k(const void* W, const float* X, float* Y, int N, int K) {
    const int bpr = K / 256;
    const char* W_bytes = static_cast<const char*>(W);
    float blk_buf[256];
    for (int r = 0; r < N; ++r) {
        double acc = 0.0;
        const char* row_ptr = W_bytes + static_cast<size_t>(r) * bpr * 144;
        for (int b = 0; b < bpr; ++b) {
            ref_dequant_q4k_block(row_ptr + b * 144, blk_buf);
            const float* x_sub = X + b * 256;
            for (int i = 0; i < 256; ++i) {
                acc += static_cast<double>(blk_buf[i]) * static_cast<double>(x_sub[i]);
            }
        }
        Y[r] = static_cast<float>(acc);
    }
}

} // namespace
} // namespace brotensor::detail::cpu::jit

#if BROTENSOR_HAS_BRASS_JIT

#include <brass/codegen/ml_fusion.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include "../jit/kernel_disk_cache.h"
#include <atomic>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace brotensor::detail::cpu::jit {

namespace {

using namespace brass;
using namespace brass::codegen;

using BlockAccFn = void (*)(const float* w_blk, const float* x_blk, float* acc8);

static void populate_gemv_q8_0_acc_block(Module& mod) {
    Function* fn = mod.create_function(
        "gemv_q8_0_acc_block",
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr()});
    KernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    kb.position_at_end(entry);
    Value* w_blk = kb.builder().add_block_param(entry, Type::ptr());
    Value* x_blk = kb.builder().add_block_param(entry, Type::ptr());
    Value* acc8  = kb.builder().add_block_param(entry, Type::ptr());

    Value* vsum = kb.vload_f32x8(acc8, 0);
    for (int c = 0; c < 4; ++c) {
        Value* vw = kb.vload_f32x8(w_blk, c * 32);
        Value* vx = kb.vload_f32x8(x_blk, c * 32);
        vsum = kb.vfma(vw, vx, vsum);
    }
    kb.vstore_f32x8(acc8, vsum, 0);
    kb.builder().build_ret_void();
}

static void populate_gemv_q4_k_acc_block(Module& mod) {
    Function* fn = mod.create_function(
        "gemv_q4_k_acc_block",
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr()});
    KernelBuilder kb(mod, fn);
    BasicBlock* entry = kb.builder().append_block("entry");
    kb.position_at_end(entry);
    Value* w_blk = kb.builder().add_block_param(entry, Type::ptr());
    Value* x_blk = kb.builder().add_block_param(entry, Type::ptr());
    Value* acc8  = kb.builder().add_block_param(entry, Type::ptr());

    Value* vsum = kb.vload_f32x8(acc8, 0);
    for (int c = 0; c < 32; ++c) {
        Value* vw = kb.vload_f32x8(w_blk, c * 32);
        Value* vx = kb.vload_f32x8(x_blk, c * 32);
        vsum = kb.vfma(vw, vx, vsum);
    }
    kb.vstore_f32x8(acc8, vsum, 0);
    kb.builder().build_ret_void();
}

static inline void fast_dequant_q8_0_block(const void* blk_ptr, float* out32) {
#if defined(__AVX2__)
    uint16_t d_raw;
    std::memcpy(&d_raw, blk_ptr, 2);
    const float d = fp16_bits_to_fp32(d_raw);
    const int8_t* qs = reinterpret_cast<const int8_t*>(static_cast<const char*>(blk_ptr) + 2);
    __m256 d_vec = _mm256_set1_ps(d);
    for (int i = 0; i < 4; ++i) {
        __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + i * 8));
        __m256i q32 = _mm256_cvtepi8_epi32(q8);
        __m256 qf = _mm256_cvtepi32_ps(q32);
        __m256 w = _mm256_mul_ps(qf, d_vec);
        _mm256_storeu_ps(out32 + i * 8, w);
    }
#else
    ref_dequant_q8_0_block(blk_ptr, out32);
#endif
}

struct QuantJitKernels {
    bool available = false;
    BlockAccFn q8_0_acc_fn = nullptr;
    BlockAccFn q4_k_acc_fn = nullptr;
};

static std::mutex s_quant_jit_mu;
static std::atomic<bool> s_quant_jit_ready{false};
static QuantJitKernels s_quant_kernels;

static const QuantJitKernels& get_quant_kernels() {
    if (s_quant_jit_ready.load(std::memory_order_acquire)) {
        return s_quant_kernels;
    }
    std::lock_guard<std::mutex> lk(s_quant_jit_mu);
    if (!s_quant_jit_ready.load(std::memory_order_relaxed)) {
        try {
            auto& cache = brotensor::jit::KernelDiskCache::instance();
            brotensor::jit::KernelDiskCache::CpuCompileOptions opts;

            s_quant_kernels.q8_0_acc_fn = cache.compile_or_load_cpu_fn<BlockAccFn>(
                "cpu_quant_gemv_q8_0_v1", "gemv_q8_0_acc_block",
                populate_gemv_q8_0_acc_block, opts);
            s_quant_kernels.q4_k_acc_fn = cache.compile_or_load_cpu_fn<BlockAccFn>(
                "cpu_quant_gemv_q4_k_v1", "gemv_q4_k_acc_block",
                populate_gemv_q4_k_acc_block, opts);

            s_quant_kernels.available =
                (s_quant_kernels.q8_0_acc_fn != nullptr && s_quant_kernels.q4_k_acc_fn != nullptr);
        } catch (const std::exception& e) {
            std::cerr << "brotensor: Brass JIT quant kernels initialization failed: " << e.what() << "\n";
            s_quant_kernels.available = false;
        }
        s_quant_jit_ready.store(true, std::memory_order_release);
    }
    return s_quant_kernels;
}

static void run_q8_0_rows(BlockAccFn acc_fn, const void* W, const float* X, float* Y, int N, int K) {
    const int bpr = K / 32;
    const char* W_bytes = static_cast<const char*>(W);
    alignas(32) float blk_buf[32];
    alignas(32) float acc8[8];
    for (int r = 0; r < N; ++r) {
        std::memset(acc8, 0, sizeof(acc8));
        const char* row_ptr = W_bytes + static_cast<size_t>(r) * bpr * 34;
        for (int b = 0; b < bpr; ++b) {
            fast_dequant_q8_0_block(row_ptr + b * 34, blk_buf);
            acc_fn(blk_buf, X + b * 32, acc8);
        }
        Y[r] = ((acc8[0] + acc8[1]) + (acc8[2] + acc8[3])) +
               ((acc8[4] + acc8[5]) + (acc8[6] + acc8[7]));
    }
}

static void run_q4_k_rows(BlockAccFn acc_fn, const void* W, const float* X, float* Y, int N, int K) {
    const int bpr = K / 256;
    const char* W_bytes = static_cast<const char*>(W);
    alignas(32) float blk_buf[256];
    alignas(32) float acc8[8];
    for (int r = 0; r < N; ++r) {
        std::memset(acc8, 0, sizeof(acc8));
        const char* row_ptr = W_bytes + static_cast<size_t>(r) * bpr * 144;
        for (int b = 0; b < bpr; ++b) {
            ref_dequant_q4k_block(row_ptr + b * 144, blk_buf);
            acc_fn(blk_buf, X + b * 256, acc8);
        }
        Y[r] = ((acc8[0] + acc8[1]) + (acc8[2] + acc8[3])) +
               ((acc8[4] + acc8[5]) + (acc8[6] + acc8[7]));
    }
}

} // namespace

void prewarm_quant_jit_kernels() {
    (void)get_quant_kernels();
}

void reset_quant_jit_caches_for_testing() {
    std::lock_guard<std::mutex> lk(s_quant_jit_mu);
    s_quant_jit_ready.store(false, std::memory_order_relaxed);
    s_quant_kernels = QuantJitKernels{};
}

void gemv_q8_0(const void* W, const float* X, float* Y, int N, int K) {
    if (N <= 0 || K <= 0) return;
    const auto& k = get_quant_kernels();
    if (k.available && k.q8_0_acc_fn != nullptr) {
        if (N >= 8) {
            const size_t row_bytes = static_cast<size_t>(K / 32) * 34;
            int n_threads = std::min<int>(detail::cpu::ThreadPool::instance().num_threads(), N);
            if (n_threads <= 1) {
                run_q8_0_rows(k.q8_0_acc_fn, W, X, Y, N, K);
            } else {
                int chunk_size = (N + n_threads - 1) / n_threads;
                int num_chunks = (N + chunk_size - 1) / chunk_size;
                detail::cpu::parallel_for(static_cast<std::size_t>(num_chunks), [&](std::size_t c) {
                    int r_start = static_cast<int>(c) * chunk_size;
                    int r_end = std::min(r_start + chunk_size, N);
                    if (r_start >= r_end) return;
                    const char* W_bytes = static_cast<const char*>(W);
                    const void* W_slice = W_bytes + static_cast<size_t>(r_start) * row_bytes;
                    float* Y_slice = Y + r_start;
                    run_q8_0_rows(k.q8_0_acc_fn, W_slice, X, Y_slice, r_end - r_start, K);
                });
            }
        } else {
            run_q8_0_rows(k.q8_0_acc_fn, W, X, Y, N, K);
        }
        return;
    }
    ref_gemv_q8_0(W, X, Y, N, K);
}

void gemv_q4_k(const void* W, const float* X, float* Y, int N, int K) {
    if (N <= 0 || K <= 0) return;
    const auto& k = get_quant_kernels();
    if (k.available && k.q4_k_acc_fn != nullptr) {
        if (N >= 8) {
            const size_t row_bytes = static_cast<size_t>(K / 256) * 144;
            int n_threads = std::min<int>(detail::cpu::ThreadPool::instance().num_threads(), N);
            if (n_threads <= 1) {
                run_q4_k_rows(k.q4_k_acc_fn, W, X, Y, N, K);
            } else {
                int chunk_size = (N + n_threads - 1) / n_threads;
                int num_chunks = (N + chunk_size - 1) / chunk_size;
                detail::cpu::parallel_for(static_cast<std::size_t>(num_chunks), [&](std::size_t c) {
                    int r_start = static_cast<int>(c) * chunk_size;
                    int r_end = std::min(r_start + chunk_size, N);
                    if (r_start >= r_end) return;
                    const char* W_bytes = static_cast<const char*>(W);
                    const void* W_slice = W_bytes + static_cast<size_t>(r_start) * row_bytes;
                    float* Y_slice = Y + r_start;
                    run_q4_k_rows(k.q4_k_acc_fn, W_slice, X, Y_slice, r_end - r_start, K);
                });
            }
        } else {
            run_q4_k_rows(k.q4_k_acc_fn, W, X, Y, N, K);
        }
        return;
    }
    ref_gemv_q4k(W, X, Y, N, K);
}

} // namespace brotensor::detail::cpu::jit

#else // !BROTENSOR_HAS_BRASS_JIT

namespace brotensor::detail::cpu::jit {

void prewarm_quant_jit_kernels() {}
void reset_quant_jit_caches_for_testing() {}

void gemv_q8_0(const void* W, const float* X, float* Y, int N, int K) {
    if (N <= 0 || K <= 0) return;
    ref_gemv_q8_0(W, X, Y, N, K);
}

void gemv_q4_k(const void* W, const float* X, float* Y, int N, int K) {
    if (N <= 0 || K <= 0) return;
    ref_gemv_q4k(W, X, Y, N, K);
}

} // namespace brotensor::detail::cpu::jit

#endif // BROTENSOR_HAS_BRASS_JIT

// ─── OpsVTable Registrations for CPU Quantization Ops ───────────────────────

namespace brotensor::detail::cpu {

namespace {

static void validate_w_q8_0(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q8_0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q8_0");
    }
    if (W.cols % 32 != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W cols must be a multiple of 32");
    }
}

static void validate_w_q4k(const Tensor& W, const char* op) {
    if (W.dtype != Dtype::Q4_K) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W must be Dtype::Q4_K");
    }
    if (W.cols % 256 != 0) {
        throw std::runtime_error(std::string("brotensor: ") + op + ": W cols must be a multiple of 256");
    }
}

static inline void local_dequant_q8_0_block(const void* blk_ptr, float* out32) {
    uint16_t d_raw;
    std::memcpy(&d_raw, blk_ptr, 2);
    const float d = fp16_bits_to_fp32(d_raw);
    const int8_t* qs = reinterpret_cast<const int8_t*>(static_cast<const char*>(blk_ptr) + 2);
    for (int i = 0; i < 32; ++i) {
        out32[i] = static_cast<float>(qs[i]) * d;
    }
}

static inline void local_dequant_q4k_block(const void* blk_ptr, float* out256) {
    const uint8_t* p = static_cast<const uint8_t*>(blk_ptr);
    uint16_t d_raw, dmin_raw;
    std::memcpy(&d_raw, p, 2);
    std::memcpy(&dmin_raw, p + 2, 2);
    const float d = fp16_bits_to_fp32(d_raw);
    const float dmin = fp16_bits_to_fp32(dmin_raw);

    const uint8_t* scales = p + 4;
    const uint8_t* qs = p + 16;

    uint8_t sc[8], m[8];
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = scales[j]     & 0x3F;
            m[j]  = scales[j + 4] & 0x3F;
        } else {
            sc[j] = static_cast<uint8_t>((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
            m[j]  = static_cast<uint8_t>((scales[j + 4] >> 4)   | ((scales[j - 0] >> 6) << 4));
        }
    }

    for (int p_idx = 0; p_idx < 4; ++p_idx) {
        const int is_lo = 2 * p_idx;
        const int is_hi = 2 * p_idx + 1;
        const float w_lo = static_cast<float>(sc[is_lo]) * d;
        const float w_hi = static_cast<float>(sc[is_hi]) * d;
        const float b_lo = static_cast<float>(m [is_lo]) * dmin;
        const float b_hi = static_cast<float>(m [is_hi]) * dmin;

        for (int l = 0; l < 32; ++l) {
            const uint8_t byte = qs[p_idx * 32 + l];
            const int n_lo = byte & 0x0F;
            const int n_hi = (byte >> 4) & 0x0F;
            out256[is_lo * 32 + l] = w_lo * static_cast<float>(n_lo) - b_lo;
            out256[is_hi * 32 + l] = w_hi * static_cast<float>(n_hi) - b_hi;
        }
    }
}

} // namespace

void dequant_q8_0_to_fp16(const Tensor& W_q8, Tensor& W_fp16) {
    validate_w_q8_0(W_q8, "dequant_q8_0_to_fp16");
    const int out = W_q8.rows;
    const int K   = W_q8.cols;
    if (W_fp16.rows != out || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(out, K, Dtype::FP16);
    }
    if (out == 0 || K == 0) return;
    const int bpr = K / 32;
    const char* src = static_cast<const char*>(W_q8.data);
    uint16_t* dst = static_cast<uint16_t*>(W_fp16.data);

    auto dequant_row = [&](int r) {
        float buf[32];
        const char* rsrc = src + static_cast<size_t>(r) * bpr * 34;
        uint16_t* rdst = dst + static_cast<size_t>(r) * K;
        for (int b = 0; b < bpr; ++b) {
            local_dequant_q8_0_block(rsrc + b * 34, buf);
            for (int i = 0; i < 32; ++i) {
                rdst[b * 32 + i] = fp32_to_fp16_bits(buf[i]);
            }
        }
    };

    if (out >= 8) {
        detail::cpu::parallel_for(static_cast<std::size_t>(out), [&](std::size_t r) {
            dequant_row(static_cast<int>(r));
        });
    } else {
        for (int r = 0; r < out; ++r) dequant_row(r);
    }
}

void dequant_q4k_to_fp16(const Tensor& W_q4k, Tensor& W_fp16) {
    validate_w_q4k(W_q4k, "dequant_q4k_to_fp16");
    const int out = W_q4k.rows;
    const int K   = W_q4k.cols;
    if (W_fp16.rows != out || W_fp16.cols != K || W_fp16.dtype != Dtype::FP16) {
        W_fp16.resize(out, K, Dtype::FP16);
    }
    if (out == 0 || K == 0) return;
    const int bpr = K / 256;
    const char* src = static_cast<const char*>(W_q4k.data);
    uint16_t* dst = static_cast<uint16_t*>(W_fp16.data);

    auto dequant_row = [&](int r) {
        float buf[256];
        const char* rsrc = src + static_cast<size_t>(r) * bpr * 144;
        uint16_t* rdst = dst + static_cast<size_t>(r) * K;
        for (int b = 0; b < bpr; ++b) {
            local_dequant_q4k_block(rsrc + b * 144, buf);
            for (int i = 0; i < 256; ++i) {
                rdst[b * 256 + i] = fp32_to_fp16_bits(buf[i]);
            }
        }
    };

    if (out >= 8) {
        detail::cpu::parallel_for(static_cast<std::size_t>(out), [&](std::size_t r) {
            dequant_row(static_cast<int>(r));
        });
    } else {
        for (int r = 0; r < out; ++r) dequant_row(r);
    }
}

void linear_forward_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                              const Tensor& x, Tensor& y) {
    validate_w_q8_0(W_q8, "linear_forward_q8_0_fp16");
    const int out = W_q8.rows;
    const int K   = W_q8.cols;
    if (x.size() != K) {
        throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: x size mismatch");
    }
    if (bias && bias->size() > 0 && bias->size() != out) {
        throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: bias size mismatch");
    }
    if (y.rows != out || y.cols != 1 || (y.dtype != Dtype::FP16 && y.dtype != Dtype::FP32)) {
        y.resize(out, 1, Dtype::FP16);
    }
    if (out == 0) return;

    std::vector<float> x_f32(K);
    if (x.dtype == Dtype::FP16) {
        const uint16_t* xp = static_cast<const uint16_t*>(x.data);
        for (int i = 0; i < K; ++i) x_f32[i] = fp16_bits_to_fp32(xp[i]);
    } else if (x.dtype == Dtype::FP32) {
        std::memcpy(x_f32.data(), x.data, K * sizeof(float));
    } else {
        throw std::runtime_error("brotensor: linear_forward_q8_0_fp16: x must be FP16 or FP32");
    }

    std::vector<float> y_f32(out);
    jit::gemv_q8_0(W_q8.data, x_f32.data(), y_f32.data(), out, K);

    if (bias && bias->size() > 0) {
        if (bias->dtype == Dtype::FP16) {
            const uint16_t* bp = static_cast<const uint16_t*>(bias->data);
            for (int i = 0; i < out; ++i) y_f32[i] += fp16_bits_to_fp32(bp[i]);
        } else if (bias->dtype == Dtype::FP32) {
            const float* bp = static_cast<const float*>(bias->data);
            for (int i = 0; i < out; ++i) y_f32[i] += bp[i];
        }
    }

    if (y.dtype == Dtype::FP16) {
        uint16_t* yp = static_cast<uint16_t*>(y.data);
        for (int i = 0; i < out; ++i) yp[i] = fp32_to_fp16_bits(y_f32[i]);
    } else {
        std::memcpy(y.data, y_f32.data(), out * sizeof(float));
    }
}

void linear_forward_batched_q8_0_fp16(const Tensor& W_q8, const Tensor* bias,
                                      const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q8_0(W_q8, "linear_forward_batched_q8_0_fp16");
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q8.rows;
    if (W_q8.cols != K) {
        throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16: shape mismatch");
    }
    if (bias && bias->size() > 0 && bias->size() != out) {
        throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16: bias size mismatch");
    }
    if (Y_BD.rows != B || Y_BD.cols != out || (Y_BD.dtype != Dtype::FP16 && Y_BD.dtype != Dtype::FP32)) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;

    for (int b = 0; b < B; ++b) {
        std::vector<float> x_f32(K);
        if (X_BD.dtype == Dtype::FP16) {
            const uint16_t* xp = static_cast<const uint16_t*>(X_BD.data) + static_cast<size_t>(b) * K;
            for (int i = 0; i < K; ++i) x_f32[i] = fp16_bits_to_fp32(xp[i]);
        } else if (X_BD.dtype == Dtype::FP32) {
            const float* xp = static_cast<const float*>(X_BD.data) + static_cast<size_t>(b) * K;
            std::memcpy(x_f32.data(), xp, K * sizeof(float));
        } else {
            throw std::runtime_error("brotensor: linear_forward_batched_q8_0_fp16: X must be FP16 or FP32");
        }

        std::vector<float> y_f32(out);
        jit::gemv_q8_0(W_q8.data, x_f32.data(), y_f32.data(), out, K);

        if (bias && bias->size() > 0) {
            if (bias->dtype == Dtype::FP16) {
                const uint16_t* bp = static_cast<const uint16_t*>(bias->data);
                for (int i = 0; i < out; ++i) y_f32[i] += fp16_bits_to_fp32(bp[i]);
            } else if (bias->dtype == Dtype::FP32) {
                const float* bp = static_cast<const float*>(bias->data);
                for (int i = 0; i < out; ++i) y_f32[i] += bp[i];
            }
        }

        if (Y_BD.dtype == Dtype::FP16) {
            uint16_t* yp = static_cast<uint16_t*>(Y_BD.data) + static_cast<size_t>(b) * out;
            for (int i = 0; i < out; ++i) yp[i] = fp32_to_fp16_bits(y_f32[i]);
        } else {
            float* yp = static_cast<float*>(Y_BD.data) + static_cast<size_t>(b) * out;
            std::memcpy(yp, y_f32.data(), out * sizeof(float));
        }
    }
}

void linear_forward_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                             const Tensor& x, Tensor& y) {
    validate_w_q4k(W_q4k, "linear_forward_q4k_fp16");
    const int out = W_q4k.rows;
    const int K   = W_q4k.cols;
    if (x.size() != K) {
        throw std::runtime_error("brotensor: linear_forward_q4k_fp16: x size mismatch");
    }
    if (bias && bias->size() > 0 && bias->size() != out) {
        throw std::runtime_error("brotensor: linear_forward_q4k_fp16: bias size mismatch");
    }
    if (y.rows != out || y.cols != 1 || (y.dtype != Dtype::FP16 && y.dtype != Dtype::FP32)) {
        y.resize(out, 1, Dtype::FP16);
    }
    if (out == 0) return;

    std::vector<float> x_f32(K);
    if (x.dtype == Dtype::FP16) {
        const uint16_t* xp = static_cast<const uint16_t*>(x.data);
        for (int i = 0; i < K; ++i) x_f32[i] = fp16_bits_to_fp32(xp[i]);
    } else if (x.dtype == Dtype::FP32) {
        std::memcpy(x_f32.data(), x.data, K * sizeof(float));
    } else {
        throw std::runtime_error("brotensor: linear_forward_q4k_fp16: x must be FP16 or FP32");
    }

    std::vector<float> y_f32(out);
    jit::gemv_q4_k(W_q4k.data, x_f32.data(), y_f32.data(), out, K);

    if (bias && bias->size() > 0) {
        if (bias->dtype == Dtype::FP16) {
            const uint16_t* bp = static_cast<const uint16_t*>(bias->data);
            for (int i = 0; i < out; ++i) y_f32[i] += fp16_bits_to_fp32(bp[i]);
        } else if (bias->dtype == Dtype::FP32) {
            const float* bp = static_cast<const float*>(bias->data);
            for (int i = 0; i < out; ++i) y_f32[i] += bp[i];
        }
    }

    if (y.dtype == Dtype::FP16) {
        uint16_t* yp = static_cast<uint16_t*>(y.data);
        for (int i = 0; i < out; ++i) yp[i] = fp32_to_fp16_bits(y_f32[i]);
    } else {
        std::memcpy(y.data, y_f32.data(), out * sizeof(float));
    }
}

void linear_forward_batched_q4k_fp16(const Tensor& W_q4k, const Tensor* bias,
                                     const Tensor& X_BD, Tensor& Y_BD) {
    validate_w_q4k(W_q4k, "linear_forward_batched_q4k_fp16");
    const int B   = X_BD.rows;
    const int K   = X_BD.cols;
    const int out = W_q4k.rows;
    if (W_q4k.cols != K) {
        throw std::runtime_error("brotensor: linear_forward_batched_q4k_fp16: shape mismatch");
    }
    if (bias && bias->size() > 0 && bias->size() != out) {
        throw std::runtime_error("brotensor: linear_forward_batched_q4k_fp16: bias size mismatch");
    }
    if (Y_BD.rows != B || Y_BD.cols != out || (Y_BD.dtype != Dtype::FP16 && Y_BD.dtype != Dtype::FP32)) {
        Y_BD.resize(B, out, Dtype::FP16);
    }
    if (B == 0 || out == 0) return;

    for (int b = 0; b < B; ++b) {
        std::vector<float> x_f32(K);
        if (X_BD.dtype == Dtype::FP16) {
            const uint16_t* xp = static_cast<const uint16_t*>(X_BD.data) + static_cast<size_t>(b) * K;
            for (int i = 0; i < K; ++i) x_f32[i] = fp16_bits_to_fp32(xp[i]);
        } else if (X_BD.dtype == Dtype::FP32) {
            const float* xp = static_cast<const float*>(X_BD.data) + static_cast<size_t>(b) * K;
            std::memcpy(x_f32.data(), xp, K * sizeof(float));
        } else {
            throw std::runtime_error("brotensor: linear_forward_batched_q4k_fp16: X must be FP16 or FP32");
        }

        std::vector<float> y_f32(out);
        jit::gemv_q4_k(W_q4k.data, x_f32.data(), y_f32.data(), out, K);

        if (bias && bias->size() > 0) {
            if (bias->dtype == Dtype::FP16) {
                const uint16_t* bp = static_cast<const uint16_t*>(bias->data);
                for (int i = 0; i < out; ++i) y_f32[i] += fp16_bits_to_fp32(bp[i]);
            } else if (bias->dtype == Dtype::FP32) {
                const float* bp = static_cast<const float*>(bias->data);
                for (int i = 0; i < out; ++i) y_f32[i] += bp[i];
            }
        }

        if (Y_BD.dtype == Dtype::FP16) {
            uint16_t* yp = static_cast<uint16_t*>(Y_BD.data) + static_cast<size_t>(b) * out;
            for (int i = 0; i < out; ++i) yp[i] = fp32_to_fp16_bits(y_f32[i]);
        } else {
            float* yp = static_cast<float*>(Y_BD.data) + static_cast<size_t>(b) * out;
            std::memcpy(yp, y_f32.data(), out * sizeof(float));
        }
    }
}

} // namespace brotensor::detail::cpu
