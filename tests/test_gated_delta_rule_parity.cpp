// CPU↔GPU parity for the brolm Qwen3-Next text-path ops:
//   * l2_norm_forward / l2_norm_backward
//   * gated_delta_rule_chunked / gated_delta_rule_step
//
// All FP32 — the CPU backend is FP32-only and the parity surface for these
// ops is FP32 on every backend.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;
using brotensor::Dtype;

namespace {

// ─── l2_norm ────────────────────────────────────────────────────────────────

void run_l2_fwd(int L, int num_heads, int head_dim, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X = Tensor::mat(L, num_heads * head_dim);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::l2_norm_forward(X, head_dim, num_heads, eps, cpu_Y);

    Tensor gX = X.to(gpu_device());
    Tensor gpu_Y;
    brotensor::l2_norm_forward(gX, head_dim, num_heads, eps, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y),
                    "l2_norm_fwd", 1e-5f, 1e-4f);
}

void run_l2_bwd(int L, int num_heads, int head_dim, float eps, uint64_t seed) {
    SplitMix64 rng(seed);
    Tensor X  = Tensor::mat(L, num_heads * head_dim);
    Tensor dY = Tensor::mat(L, num_heads * head_dim);
    fill_random(X,  rng);
    fill_random(dY, rng);

    Tensor cpu_dX;
    brotensor::l2_norm_backward(X, head_dim, num_heads, eps, dY, cpu_dX);

    Tensor gX  = X.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gpu_dX;
    brotensor::l2_norm_backward(gX, head_dim, num_heads, eps, gdY, gpu_dX);

    compare_tensors(cpu_dX, download_to_host(gpu_dX),
                    "l2_norm_bwd_dX", 1e-5f, 1e-4f);
}

// ─── gated_delta_rule ───────────────────────────────────────────────────────

struct GdrInputs {
    Tensor Q, K, V, a_raw, beta, log_A, state0;
};

GdrInputs make_inputs(int L, int num_heads, int d_k, int d_v, uint64_t seed) {
    SplitMix64 rng(seed);
    GdrInputs in;
    in.Q     = Tensor::mat(L, num_heads * d_k);
    in.K     = Tensor::mat(L, num_heads * d_k);
    in.V     = Tensor::mat(L, num_heads * d_v);
    in.a_raw = Tensor::mat(L, num_heads);
    in.beta  = Tensor::mat(L, num_heads);
    in.log_A = Tensor::mat(num_heads, 1);
    in.state0 = Tensor::mat(num_heads, d_v * d_k);
    fill_random(in.Q,      rng);
    fill_random(in.K,      rng);
    fill_random(in.V,      rng);
    fill_random(in.a_raw,  rng);
    fill_random(in.beta,   rng);
    fill_random(in.log_A,  rng, 0.5f);   // keep exp(log_A) modest
    // state0 stays at random non-zero so the recurrence starts from a real
    // non-trivial S_{-1} and exercises the alpha decay on every iteration.
    fill_random(in.state0, rng, 0.1f);
    return in;
}

void run_gdr_chunked(int L, int num_heads, int d_k, int d_v, uint64_t seed) {
    GdrInputs in = make_inputs(L, num_heads, d_k, d_v, seed);

    Tensor cpu_state = in.state0;
    Tensor cpu_O;
    brotensor::gated_delta_rule_chunked(in.Q, in.K, in.V, in.a_raw, in.beta,
                                        in.log_A, num_heads, d_k, d_v,
                                        cpu_state, cpu_O);

    Tensor gQ     = in.Q.to(gpu_device());
    Tensor gK     = in.K.to(gpu_device());
    Tensor gV     = in.V.to(gpu_device());
    Tensor gA     = in.a_raw.to(gpu_device());
    Tensor gB     = in.beta.to(gpu_device());
    Tensor gLA    = in.log_A.to(gpu_device());
    Tensor gpu_state = in.state0.to(gpu_device());
    Tensor gpu_O;
    brotensor::gated_delta_rule_chunked(gQ, gK, gV, gA, gB, gLA,
                                        num_heads, d_k, d_v,
                                        gpu_state, gpu_O);

    compare_tensors(cpu_O, download_to_host(gpu_O),
                    "gdr_chunked_O", 1e-4f, 1e-3f);
    compare_tensors(cpu_state, download_to_host(gpu_state),
                    "gdr_chunked_state", 1e-4f, 1e-3f);
}

// One-token "step" path with the same scan should still match CPU.
void run_gdr_step(int L, int num_heads, int d_k, int d_v, uint64_t seed) {
    GdrInputs in = make_inputs(L, num_heads, d_k, d_v, seed);

    Tensor cpu_state = in.state0;
    Tensor cpu_O;
    brotensor::gated_delta_rule_step(in.Q, in.K, in.V, in.a_raw, in.beta,
                                     in.log_A, num_heads, d_k, d_v,
                                     cpu_state, cpu_O);

    Tensor gQ     = in.Q.to(gpu_device());
    Tensor gK     = in.K.to(gpu_device());
    Tensor gV     = in.V.to(gpu_device());
    Tensor gA     = in.a_raw.to(gpu_device());
    Tensor gB     = in.beta.to(gpu_device());
    Tensor gLA    = in.log_A.to(gpu_device());
    Tensor gpu_state = in.state0.to(gpu_device());
    Tensor gpu_O;
    brotensor::gated_delta_rule_step(gQ, gK, gV, gA, gB, gLA,
                                     num_heads, d_k, d_v,
                                     gpu_state, gpu_O);

    compare_tensors(cpu_O, download_to_host(gpu_O),
                    "gdr_step_O", 1e-4f, 1e-3f);
    compare_tensors(cpu_state, download_to_host(gpu_state),
                    "gdr_step_state", 1e-4f, 1e-3f);
}

} // namespace

// ─── l2_norm ───────────────────────────────────────────────────────────────
BT_PARITY_TEST(l2_norm_fwd_tiny)  { run_l2_fwd(1, 1, 4,    1e-6f, 0x9100ull); }
BT_PARITY_TEST(l2_norm_fwd_small) { run_l2_fwd(4, 2, 16,   1e-6f, 0x9101ull); }
BT_PARITY_TEST(l2_norm_fwd_wide)  { run_l2_fwd(3, 4, 128,  1e-6f, 0x9102ull); }
BT_PARITY_TEST(l2_norm_fwd_odd)   { run_l2_fwd(5, 3, 17,   1e-6f, 0x9103ull); }
BT_PARITY_TEST(l2_norm_bwd_tiny)  { run_l2_bwd(1, 1, 4,    1e-6f, 0x9110ull); }
BT_PARITY_TEST(l2_norm_bwd_small) { run_l2_bwd(4, 2, 16,   1e-6f, 0x9111ull); }
BT_PARITY_TEST(l2_norm_bwd_wide)  { run_l2_bwd(3, 4, 128,  1e-6f, 0x9112ull); }
BT_PARITY_TEST(l2_norm_bwd_odd)   { run_l2_bwd(5, 3, 17,   1e-6f, 0x9113ull); }

// ─── gated_delta_rule ──────────────────────────────────────────────────────
BT_PARITY_TEST(gdr_chunked_tiny)  { run_gdr_chunked(2, 1, 4,  4,  0x9200ull); }
BT_PARITY_TEST(gdr_chunked_small) { run_gdr_chunked(6, 2, 8,  8,  0x9201ull); }
BT_PARITY_TEST(gdr_chunked_uneven){ run_gdr_chunked(5, 3, 7,  5,  0x9202ull); }
BT_PARITY_TEST(gdr_chunked_wide)  { run_gdr_chunked(4, 2, 32, 32, 0x9203ull); }
BT_PARITY_TEST(gdr_step_tiny)     { run_gdr_step(2, 1, 4,  4,  0x9210ull); }
BT_PARITY_TEST(gdr_step_small)    { run_gdr_step(6, 2, 8,  8,  0x9211ull); }
BT_PARITY_TEST(gdr_step_uneven)   { run_gdr_step(5, 3, 7,  5,  0x9212ull); }

int main() { return run_all("gated_delta_rule cpu/gpu parity"); }
