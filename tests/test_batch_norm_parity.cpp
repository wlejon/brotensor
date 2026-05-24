// CPU↔GPU parity tests for batch_norm_{forward,inference,backward}.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Dtype;

namespace {

// BN reductions accumulate float roundoff over M = N*H*W elements per channel.
constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-4f;

Tensor make_chan(int C, float fill) {
    Tensor t;
    t.resize(C, 1, Dtype::FP32);
    for (int i = 0; i < C; ++i) t.host_f32_mut()[i] = fill;
    return t;
}

void run_bn_forward(int N, int C, int H, int W, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);

    // gamma/beta random; running stats start the same on both sides so the
    // post-call values can be compared.
    Tensor gamma = Tensor::mat(C, 1);
    Tensor beta  = Tensor::mat(C, 1);
    fill_random(gamma, rng, 1.0f);
    fill_random(beta,  rng, 1.0f);
    Tensor run_mean_cpu = make_chan(C, 0.1f);
    Tensor run_var_cpu  = make_chan(C, 0.9f);
    Tensor run_mean_gpu = run_mean_cpu.clone().to(gpu_device());
    Tensor run_var_gpu  = run_var_cpu.clone().to(gpu_device());

    const float eps = 1e-5f;
    const float momentum = 0.1f;

    Tensor cpu_Y, cpu_sm, cpu_sr;
    brotensor::batch_norm_forward(X, gamma, beta,
                                  run_mean_cpu, run_var_cpu,
                                  N, C, H, W, eps, momentum,
                                  cpu_Y, cpu_sm, cpu_sr);

    Tensor gX  = X.to(gpu_device());
    Tensor gg  = gamma.to(gpu_device());
    Tensor gb  = beta.to(gpu_device());
    Tensor gY, gsm, gsr;
    brotensor::batch_norm_forward(gX, gg, gb,
                                  run_mean_gpu, run_var_gpu,
                                  N, C, H, W, eps, momentum,
                                  gY, gsm, gsr);

    compare_tensors(cpu_Y,        download_to_host(gY),           "bn_fwd_Y",       kAtol, kRtol);
    compare_tensors(cpu_sm,       download_to_host(gsm),          "bn_fwd_sm",      kAtol, kRtol);
    compare_tensors(cpu_sr,       download_to_host(gsr),          "bn_fwd_sr",      kAtol, kRtol);
    compare_tensors(run_mean_cpu, download_to_host(run_mean_gpu), "bn_fwd_run_mean",kAtol, kRtol);
    compare_tensors(run_var_cpu,  download_to_host(run_var_gpu),  "bn_fwd_run_var", kAtol, kRtol);
}

void run_bn_inference(int N, int C, int H, int W, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);
    Tensor gamma = Tensor::mat(C, 1);
    Tensor beta  = Tensor::mat(C, 1);
    fill_random(gamma, rng, 1.0f);
    fill_random(beta,  rng, 1.0f);
    Tensor run_mean = Tensor::mat(C, 1);
    Tensor run_var  = Tensor::mat(C, 1);
    fill_random(run_mean, rng, 0.5f);
    // running_var must be positive; shift random into [0.1, 1.1].
    for (int i = 0; i < C; ++i) {
        run_var.host_f32_mut()[i] = std::fabs(rng.next_unit()) + 0.1f;
    }
    const float eps = 1e-5f;

    Tensor cpu_Y;
    brotensor::batch_norm_inference(X, gamma, beta, run_mean, run_var,
                                    N, C, H, W, eps, cpu_Y);
    Tensor gX  = X.to(gpu_device());
    Tensor gg  = gamma.to(gpu_device());
    Tensor gb  = beta.to(gpu_device());
    Tensor grm = run_mean.to(gpu_device());
    Tensor grv = run_var.to(gpu_device());
    Tensor gY;
    brotensor::batch_norm_inference(gX, gg, gb, grm, grv,
                                    N, C, H, W, eps, gY);
    compare_tensors(cpu_Y, download_to_host(gY), "bn_inf_Y", kAtol, kRtol);
}

void run_bn_backward(int N, int C, int H, int W, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(N, C * H * W);
    fill_random(X, rng, 1.0f);
    Tensor gamma = Tensor::mat(C, 1);
    Tensor beta  = Tensor::mat(C, 1);
    fill_random(gamma, rng, 1.0f);
    fill_random(beta,  rng, 1.0f);
    Tensor run_mean = make_chan(C, 0.0f);
    Tensor run_var  = make_chan(C, 1.0f);

    // Run forward on CPU to get saved_mean / saved_rstd.
    Tensor Y, sm, sr;
    brotensor::batch_norm_forward(X, gamma, beta, run_mean, run_var,
                                  N, C, H, W, 1e-5f, 0.0f,
                                  Y, sm, sr);

    Tensor dY = Tensor::mat(N, C * H * W);
    fill_random(dY, rng, 1.0f);

    // CPU backward — caller zeros dGamma, dBeta.
    Tensor cpu_dX;
    Tensor cpu_dG = make_chan(C, 0.0f);
    Tensor cpu_dB = make_chan(C, 0.0f);
    brotensor::batch_norm_backward(X, gamma, sm, sr, dY,
                                   N, C, H, W,
                                   cpu_dX, cpu_dG, cpu_dB);

    Tensor gX  = X.to(gpu_device());
    Tensor gg  = gamma.to(gpu_device());
    Tensor gsm = sm.to(gpu_device());
    Tensor gsr = sr.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gdG = make_chan(C, 0.0f).to(gpu_device());
    Tensor gdB = make_chan(C, 0.0f).to(gpu_device());
    Tensor gdX;
    brotensor::batch_norm_backward(gX, gg, gsm, gsr, gdY,
                                   N, C, H, W,
                                   gdX, gdG, gdB);

    compare_tensors(cpu_dX, download_to_host(gdX), "bn_bwd_dX",     kAtol, kRtol);
    compare_tensors(cpu_dG, download_to_host(gdG), "bn_bwd_dGamma", kAtol, kRtol);
    compare_tensors(cpu_dB, download_to_host(gdB), "bn_bwd_dBeta",  kAtol, kRtol);
}

} // namespace

BT_PARITY_TEST(bn_fwd_small)      { run_bn_forward(2, 4, 5, 6, 0xBA00ull); }
BT_PARITY_TEST(bn_fwd_singleton)  { run_bn_forward(1, 3, 4, 4, 0xBA01ull); }
BT_PARITY_TEST(bn_inf_small)      { run_bn_inference(2, 4, 5, 6, 0xBA10ull); }
BT_PARITY_TEST(bn_inf_singleton)  { run_bn_inference(1, 3, 4, 4, 0xBA11ull); }
BT_PARITY_TEST(bn_bwd_small)      { run_bn_backward(2, 4, 5, 6, 0xBA20ull); }
BT_PARITY_TEST(bn_bwd_bigger)     { run_bn_backward(3, 8, 7, 9, 0xBA21ull); }

int main() { return run_all("batch_norm cpu/gpu parity"); }
