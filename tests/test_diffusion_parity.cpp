// CPU↔GPU parity tests for the diffusion sampler ops (CHUNK 4).
//
//   ddim_step           — x_prev OVERWRITTEN.
//   euler_step          — x_prev OVERWRITTEN.
//   dpmpp_2m_step       — x_prev + x0_out OVERWRITTEN.
//   timestep_embedding  — Y OVERWRITTEN.
//
// DTYPE NOTE: the GPU sampler kernels (ddim/euler/dpmpp_2m) run FP16 — their
// tensors MUST be FP16. The CPU backend is FP32-only. So for these three ops
// the parity harness:
//   * quantises the random inputs through FP16 (so CPU and GPU see the SAME
//     input values),
//   * feeds FP16 tensors to the GPU and FP32 tensors to the CPU,
//   * compares with a loose FP16-driven tolerance (atol/rtol 1e-2).
// The internal arithmetic is identical FP32 math on both backends; the gap is
// purely FP16 input/output rounding on the GPU side.
//
// timestep_embedding is FP32 on BOTH backends, so it gets a tighter tolerance
// (relaxed only slightly for the exp/sin/cos fast-math intrinsics).

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

// Quantise a value through FP16 so CPU (FP32) and GPU (FP16) start from the
// exact same input bit pattern.
inline float q16(float v) {
    return brotensor::fp16_bits_to_fp32(brotensor::fp32_to_fp16_bits(v));
}

// Build a CPU FP32 tensor whose values are FP16-quantised randoms.
Tensor make_q16_cpu(int rows, int cols, SplitMix64& rng, float scale) {
    Tensor t = Tensor::mat(rows, cols);
    for (int i = 0; i < t.size(); ++i) t.ptr()[i] = q16(rng.next_unit() * scale);
    return t;
}

// Upload a CPU FP32 tensor's values as an FP16 CUDA tensor of the same shape.
Tensor to_fp16_cuda(const Tensor& cpu) {
    const int n = cpu.size();
    std::vector<uint16_t> h(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h[i] = brotensor::fp32_to_fp16_bits(cpu[i]);
    return Tensor::from_host_fp16_on(Device::CUDA, h.data(),
                                     cpu.rows, cpu.cols);
}

// Download an FP16 CUDA tensor into a CPU FP32 tensor.
Tensor fp16_cuda_to_cpu(const Tensor& g) {
    brotensor::sync_all();
    std::vector<uint16_t> h = g.to_host_vector_fp16();
    Tensor out = Tensor::mat(g.rows, g.cols);
    for (int i = 0; i < out.size(); ++i)
        out.ptr()[i] = brotensor::fp16_bits_to_fp32(h[i]);
    return out;
}

// ─── ddim_step ─────────────────────────────────────────────────────────────
void run_ddim(int rows, int cols, float alpha_t, float alpha_prev,
              float sigma_t, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x_t   = make_q16_cpu(rows, cols, rng, 1.0f);
    Tensor eps   = make_q16_cpu(rows, cols, rng, 1.0f);

    Tensor cpu_xp;
    brotensor::ddim_step(x_t, eps, alpha_t, alpha_prev, sigma_t, cpu_xp);

    Tensor gx = to_fp16_cuda(x_t);
    Tensor ge = to_fp16_cuda(eps);
    Tensor gpu_xp;
    brotensor::ddim_step(gx, ge, alpha_t, alpha_prev, sigma_t, gpu_xp);

    compare_tensors(cpu_xp, fp16_cuda_to_cpu(gpu_xp), "ddim_step",
                    1e-2f, 1e-2f);
}

// ─── euler_step ────────────────────────────────────────────────────────────
void run_euler(int rows, int cols, float sigma_t, float sigma_prev,
               uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x_t = make_q16_cpu(rows, cols, rng, 1.0f);
    Tensor eps = make_q16_cpu(rows, cols, rng, 1.0f);

    Tensor cpu_xp;
    brotensor::euler_step(x_t, eps, sigma_t, sigma_prev, cpu_xp);

    Tensor gx = to_fp16_cuda(x_t);
    Tensor ge = to_fp16_cuda(eps);
    Tensor gpu_xp;
    brotensor::euler_step(gx, ge, sigma_t, sigma_prev, gpu_xp);

    compare_tensors(cpu_xp, fp16_cuda_to_cpu(gpu_xp), "euler_step",
                    1e-2f, 1e-2f);
}

// ─── dpmpp_2m_step — two outputs ───────────────────────────────────────────
void run_dpmpp(int rows, int cols, float sigma_t,
               float c_xt, float c_x0t, float c_x0prev, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor x_t  = make_q16_cpu(rows, cols, rng, 1.0f);
    Tensor eps  = make_q16_cpu(rows, cols, rng, 1.0f);
    Tensor x0p  = make_q16_cpu(rows, cols, rng, 1.0f);

    Tensor cpu_xp, cpu_x0;
    brotensor::dpmpp_2m_step(x_t, eps, x0p, sigma_t,
                             c_xt, c_x0t, c_x0prev, cpu_xp, cpu_x0);

    Tensor gx  = to_fp16_cuda(x_t);
    Tensor ge  = to_fp16_cuda(eps);
    Tensor gx0 = to_fp16_cuda(x0p);
    Tensor gpu_xp, gpu_x0;
    brotensor::dpmpp_2m_step(gx, ge, gx0, sigma_t,
                             c_xt, c_x0t, c_x0prev, gpu_xp, gpu_x0);

    compare_tensors(cpu_xp, fp16_cuda_to_cpu(gpu_xp), "dpmpp_x_prev",
                    1e-2f, 1e-2f);
    compare_tensors(cpu_x0, fp16_cuda_to_cpu(gpu_x0), "dpmpp_x0_out",
                    1e-2f, 1e-2f);
}

// ─── timestep_embedding — FP32 on both backends ────────────────────────────
void run_timestep(int N, int dim, float max_period, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor ts = Tensor::vec(N);
    for (int i = 0; i < N; ++i) ts.ptr()[i] = (rng.next_f01()) * 1000.0f;

    Tensor cpu_Y;
    brotensor::timestep_embedding(ts, dim, max_period, cpu_Y);

    Tensor gts = ts.to(Device::CUDA);
    Tensor gpu_Y;
    brotensor::timestep_embedding(gts, dim, max_period, gpu_Y);

    // FP32 both sides; relaxed only for exp/sin/cos fast-math intrinsics.
    compare_tensors(cpu_Y, download_to_host(gpu_Y), "timestep_embedding",
                    1e-4f, 1e-3f);
}

} // namespace

// ─── ddim_step ─────────────────────────────────────────────────────────────
BT_PARITY_TEST(diffusion_ddim_small) {
    run_ddim(4, 8, 0.7f, 0.85f, 0.1f, 0x7200ull);
}
BT_PARITY_TEST(diffusion_ddim_wide) {
    run_ddim(2, 64, 0.45f, 0.6f, 0.2f, 0x7201ull);
}
BT_PARITY_TEST(diffusion_ddim_zero_sigma) {
    run_ddim(3, 16, 0.9f, 0.95f, 0.0f, 0x7202ull);   // deterministic DDIM
}
BT_PARITY_TEST(diffusion_ddim_late_step) {
    run_ddim(2, 32, 0.02f, 0.05f, 0.05f, 0x7203ull);
}

// ─── euler_step ────────────────────────────────────────────────────────────
BT_PARITY_TEST(diffusion_euler_small) {
    run_euler(4, 8, 5.0f, 3.0f, 0x7210ull);
}
BT_PARITY_TEST(diffusion_euler_wide) {
    run_euler(2, 64, 10.0f, 7.5f, 0x7211ull);
}
BT_PARITY_TEST(diffusion_euler_final) {
    run_euler(3, 16, 1.0f, 0.0f, 0x7212ull);   // sigma_prev = 0 last step
}

// ─── dpmpp_2m_step ─────────────────────────────────────────────────────────
BT_PARITY_TEST(diffusion_dpmpp_small) {
    run_dpmpp(4, 8, 4.0f, 0.6f, 0.5f, -0.1f, 0x7220ull);
}
BT_PARITY_TEST(diffusion_dpmpp_wide) {
    run_dpmpp(2, 64, 8.0f, 0.5f, 0.7f, -0.2f, 0x7221ull);
}
BT_PARITY_TEST(diffusion_dpmpp_neg_coef) {
    run_dpmpp(3, 32, 2.5f, 0.55f, 0.65f, -0.2f, 0x7222ull);
}

// ─── timestep_embedding ────────────────────────────────────────────────────
BT_PARITY_TEST(diffusion_timestep_even) {
    run_timestep(5, 320, 10000.0f, 0x7230ull);   // SDXL-style dim
}
BT_PARITY_TEST(diffusion_timestep_small_even) {
    run_timestep(3, 16, 10000.0f, 0x7231ull);
}
BT_PARITY_TEST(diffusion_timestep_odd) {
    run_timestep(4, 17, 10000.0f, 0x7232ull);   // odd dim — tail slot zeroed
}
BT_PARITY_TEST(diffusion_timestep_single) {
    run_timestep(1, 256, 10000.0f, 0x7233ull);
}

int main() { return run_all("diffusion sampler cpu/gpu parity"); }
