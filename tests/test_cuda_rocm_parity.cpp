// Cross-device numerical validation: NVIDIA CUDA (RTX 3060) vs AMD ROCm/HIP (Radeon 8060S gfx1151).
// Loads reference tensors produced by PyTorch on NVIDIA CUDA, executes the identical operations on AMD GPU
// via brotensor (Device::HIP), and asserts strict tolerance.
//
// The golden file (cuda_ops_ref_interleaved.safetensors) is generated on an NVIDIA machine and is not in the
// repo. Its path comes from argv[1], else the BROTENSOR_CUDA_ROCM_GOLDEN environment variable; with neither,
// or with no HIP device, the test exits kSkip, which ctest reports as skipped (SKIP_RETURN_CODE).

#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using brotensor::Device;
using brotensor::Dtype;
using brotensor::Tensor;
using brotensor::safetensors::File;
using brotensor::safetensors::TensorView;

namespace {

struct Stats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rel_max = 0.0;
};

Stats compare_fp32(const float* a, const float* b, std::size_t n) {
    Stats s;
    double sum = 0.0;
    double b_range = 1e-7;
    float b_min = b[0], b_max = b[0];
    for (std::size_t i = 0; i < n; ++i) {
        b_min = std::min(b_min, b[i]);
        b_max = std::max(b_max, b[i]);
    }
    b_range = std::max((double)(b_max - b_min), 1e-7);

    for (std::size_t i = 0; i < n; ++i) {
        double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        s.max_abs = std::max(s.max_abs, diff);
        sum += diff;
    }
    s.mean_abs = n ? sum / (double)n : 0.0;
    s.rel_max = s.max_abs / b_range;
    return s;
}

Stats compare_fp16(const uint16_t* a, const uint16_t* b, std::size_t n) {
    std::vector<float> af(n), bf(n);
    for (std::size_t i = 0; i < n; ++i) {
        af[i] = brotensor::fp16_bits_to_fp32(a[i]);
        bf[i] = brotensor::fp16_bits_to_fp32(b[i]);
    }
    return compare_fp32(af.data(), bf.data(), n);
}

int failures = 0;

void verify_op(const char* name, const Stats& s, double max_tol, double rel_tol) {
    bool ok = (s.max_abs <= max_tol) || (s.rel_max <= rel_tol);
    std::printf("  [%s] %-25s : max_abs = %9.3e, mean_abs = %9.3e, rel_max = %9.3e (limit %9.3e)\n",
                ok ? "PASS" : "FAIL", name, s.max_abs, s.mean_abs, s.rel_max, max_tol);
    if (!ok) {
        std::printf("    --> FAILED tolerance check: max_abs %.3e > %.3e and rel_max %.3e > %.3e\n",
                    s.max_abs, max_tol, s.rel_max, rel_tol);
        failures++;
    }
}

Tensor load_tensor(const File& file, const char* name, int rows, int cols) {
    const TensorView* view = file.find(name);
    if (!view) {
        std::fprintf(stderr, "Missing tensor '%s' in safetensors file\n", name);
        std::exit(1);
    }
    Tensor t;
    brotensor::safetensors::upload(*view, rows, cols, t);
    return t.to(Device::CPU);
}

} // namespace

constexpr int kSkip = 77;

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : std::getenv("BROTENSOR_CUDA_ROCM_GOLDEN");
    if (path == nullptr || *path == '\0') {
        std::printf("SKIP: no golden file (pass a path or set BROTENSOR_CUDA_ROCM_GOLDEN)\n");
        return kSkip;
    }
    std::printf("=======================================================================\n");
    std::printf("  AMD ROCm/HIP (gfx1151) vs NVIDIA CUDA (RTX 3060) Numerical Parity   \n");
    std::printf("  Loading golden reference tensors from: %s\n", path);
    std::printf("=======================================================================\n");

    brotensor::init();
    if (!brotensor::is_available(Device::HIP)) {
        std::printf("SKIP: Device::HIP is not available on this system\n");
        return kSkip;
    }

    File f;
    try {
        f = File::open(path);
    } catch (const std::exception& e) {
        std::printf("ERROR: Failed to open %s: %s\n", path, e.what());
        return 1;
    }

    Device hip = Device::hip(0);

    // ─────────────────────────────────────────────────────────────────────────
    // 1. GEMM FP32 (M=128, K=256, N=128)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor A = load_tensor(f, "gemm_fp32_A", 128, 256).to(hip);
        Tensor B = load_tensor(f, "gemm_fp32_B", 256, 128).to(hip);
        Tensor ref = load_tensor(f, "gemm_fp32_C", 128, 128);

        Tensor C;
        brotensor::matmul(A, B, C);
        brotensor::sync(hip);

        Tensor C_host = C.to(Device::CPU);
        Stats s = compare_fp32(C_host.host_f32(), ref.host_f32(), 128 * 128);
        verify_op("GEMM FP32 (128x256 @ 256x128)", s, 1e-4, 1e-4);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. GEMM FP16 (M=128, K=256, N=128)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor A = load_tensor(f, "gemm_fp16_A", 128, 256).to(hip);
        Tensor B = load_tensor(f, "gemm_fp16_B", 256, 128).to(hip);
        Tensor ref = load_tensor(f, "gemm_fp16_C", 128, 128);

        Tensor C;
        brotensor::matmul(A, B, C);
        brotensor::sync(hip);

        Tensor C_host = C.to(Device::CPU);
        auto ref_v = ref.to_host_vector_fp16();
        auto got_v = C_host.to_host_vector_fp16();
        Stats s = compare_fp16(got_v.data(), ref_v.data(), 128 * 128);
        verify_op("GEMM FP16 (128x256 @ 256x128)", s, 1e-2, 1e-2);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. RMSNorm FP32 (32x512)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor X = load_tensor(f, "rmsnorm_X", 32, 512).to(hip);
        Tensor gamma = load_tensor(f, "rmsnorm_gamma", 512, 1).to(hip);
        Tensor ref = load_tensor(f, "rmsnorm_out_f32", 32, 512);

        Tensor Y;
        brotensor::rms_norm_forward(X, gamma, 1e-5f, Y);
        brotensor::sync(hip);

        Tensor Y_host = Y.to(Device::CPU);
        Stats s = compare_fp32(Y_host.host_f32(), ref.host_f32(), 32 * 512);
        verify_op("RMSNorm FP32 (32x512)", s, 1e-4, 1e-4);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. RMSNorm FP16 (32x512)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor X_f32 = load_tensor(f, "rmsnorm_X", 32, 512);
        Tensor gamma_f32 = load_tensor(f, "rmsnorm_gamma", 512, 1);
        Tensor ref = load_tensor(f, "rmsnorm_out_f16", 32, 512);

        Tensor X16, gamma16;
        brotensor::cast(X_f32, X16, Dtype::FP16);
        brotensor::cast(gamma_f32, gamma16, Dtype::FP16);

        Tensor X_hip = X16.to(hip);
        Tensor gamma_hip = gamma16.to(hip);
        Tensor Y;
        brotensor::rms_norm_forward(X_hip, gamma_hip, 1e-5f, Y);
        brotensor::sync(hip);

        Tensor Y_host = Y.to(Device::CPU);
        auto ref_v = ref.to_host_vector_fp16();
        auto got_v = Y_host.to_host_vector_fp16();
        Stats s = compare_fp16(got_v.data(), ref_v.data(), 32 * 512);
        verify_op("RMSNorm FP16 (32x512)", s, 5e-3, 5e-3);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 5. LayerNorm FP32 (32x512)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor X = load_tensor(f, "rmsnorm_X", 32, 512).to(hip);
        Tensor gamma = load_tensor(f, "rmsnorm_gamma", 512, 1).to(hip);
        Tensor beta = load_tensor(f, "layernorm_beta", 512, 1).to(hip);
        Tensor ref = load_tensor(f, "layernorm_out_f32", 32, 512);

        Tensor Y;
        brotensor::layernorm_forward_inference_batched(X, gamma, beta, Y, 1e-5f);
        brotensor::sync(hip);

        Tensor Y_host = Y.to(Device::CPU);
        Stats s = compare_fp32(Y_host.host_f32(), ref.host_f32(), 32 * 512);
        verify_op("LayerNorm FP32 (32x512)", s, 1e-4, 1e-4);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 6. LayerNorm FP16 (32x512)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor X_f32 = load_tensor(f, "rmsnorm_X", 32, 512);
        Tensor gamma_f32 = load_tensor(f, "rmsnorm_gamma", 512, 1);
        Tensor beta_f32 = load_tensor(f, "layernorm_beta", 512, 1);
        Tensor ref = load_tensor(f, "layernorm_out_f16", 32, 512);

        Tensor X16, gamma16, beta16;
        brotensor::cast(X_f32, X16, Dtype::FP16);
        brotensor::cast(gamma_f32, gamma16, Dtype::FP16);
        brotensor::cast(beta_f32, beta16, Dtype::FP16);

        Tensor X_hip = X16.to(hip);
        Tensor gamma_hip = gamma16.to(hip);
        Tensor beta_hip = beta16.to(hip);
        Tensor Y;
        brotensor::layernorm_forward_inference_batched(X_hip, gamma_hip, beta_hip, Y, 1e-5f);
        brotensor::sync(hip);

        Tensor Y_host = Y.to(Device::CPU);
        auto ref_v = ref.to_host_vector_fp16();
        auto got_v = Y_host.to_host_vector_fp16();
        Stats s = compare_fp16(got_v.data(), ref_v.data(), 32 * 512);
        verify_op("LayerNorm FP16 (32x512)", s, 5e-3, 5e-3);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 7. Softmax FP32 (16x1024)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor X = load_tensor(f, "softmax_in", 16, 1024).to(hip);
        Tensor ref = load_tensor(f, "softmax_out", 16, 1024);

        Tensor Y;
        brotensor::softmax_rows_forward(X, Y, 16, 1024);
        brotensor::sync(hip);

        Tensor Y_host = Y.to(Device::CPU);
        Stats s = compare_fp32(Y_host.host_f32(), ref.host_f32(), 16 * 1024);
        verify_op("Softmax FP32 (16x1024)", s, 1e-4, 1e-4);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 8. SwiGLU FP16 (64x1024 -> 64x512)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor X = load_tensor(f, "swiglu_in", 64, 1024).to(hip);
        Tensor ref = load_tensor(f, "swiglu_out", 64, 512);

        Tensor Y;
        brotensor::swiglu_forward(X, Y);
        brotensor::sync(hip);

        Tensor Y_host = Y.to(Device::CPU);
        auto ref_v = ref.to_host_vector_fp16();
        auto got_v = Y_host.to_host_vector_fp16();
        Stats s = compare_fp16(got_v.data(), ref_v.data(), 64 * 512);
        verify_op("SwiGLU FP16 (64x1024)", s, 5e-3, 5e-3);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 9. Multi-Head Attention FP16 (Seq=256, Heads=6, HeadDim=64 -> 256x384)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor Q = load_tensor(f, "attn_Q", 256, 384).to(hip);
        Tensor K = load_tensor(f, "attn_K", 256, 384).to(hip);
        Tensor V = load_tensor(f, "attn_V", 256, 384).to(hip);
        Tensor ref = load_tensor(f, "attn_out", 256, 384);

        Tensor O;
        brotensor::flash_attention_forward(Q, K, V, nullptr, 6, false, O);
        brotensor::sync(hip);

        Tensor O_host = O.to(Device::CPU);
        auto ref_v = ref.to_host_vector_fp16();
        auto got_v = O_host.to_host_vector_fp16();
        Stats s = compare_fp16(got_v.data(), ref_v.data(), 256 * 384);
        verify_op("MHA FP16 (256x384, H=6, D=64)", s, 1.5e-2, 1.5e-2);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 10. Causal Flash Attention FP16 (Seq=128, Heads=4, HeadDim=64 -> 128x256)
    // ─────────────────────────────────────────────────────────────────────────
    {
        Tensor Q = load_tensor(f, "causal_Q", 128, 256).to(hip);
        Tensor K = load_tensor(f, "causal_K", 128, 256).to(hip);
        Tensor V = load_tensor(f, "causal_V", 128, 256).to(hip);
        Tensor ref = load_tensor(f, "causal_out", 128, 256);

        Tensor O;
        brotensor::flash_attention_forward(Q, K, V, nullptr, 4, true, O);
        brotensor::sync(hip);

        Tensor O_host = O.to(Device::CPU);
        auto ref_v = ref.to_host_vector_fp16();
        auto got_v = O_host.to_host_vector_fp16();
        Stats s = compare_fp16(got_v.data(), ref_v.data(), 128 * 256);
        verify_op("Causal Flash FP16 (128x256, H=4)", s, 1.5e-2, 1.5e-2);
    }

    std::printf("=======================================================================\n");
    if (failures == 0) {
        std::printf("ALL 10 FOUNDATION OPS PASSED NUMERICAL PARITY (CUDA RTX 3060 vs ROCm 8060S)!\n");
        return 0;
    } else {
        std::printf("FAILED: %d op(s) failed numerical tolerance!\n", failures);
        return 1;
    }
}
