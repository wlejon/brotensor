// CPU↔GPU parity tests for the Qwen3-Next gated delta rule (FP32).
//
// Shapes follow src/cpu/gated_delta_rule.cpp:
//   Q, K     : (L, num_heads * d_k)
//   V        : (L, num_heads * d_v)
//   a_raw    : (L, num_heads)
//   beta     : (L, num_heads)
//   log_A    : (num_heads, 1)
//   state    : (num_heads, d_v * d_k)        — read/written in place
//   O        : (L, num_heads * d_v)
//
// Both `gated_delta_rule_chunked` and `gated_delta_rule_step` share the same
// op contract — chunked = prefill (L>1), step = single-token decode (L==1).
// We test both. State is the same baseline (random non-zero) on CPU and GPU
// so the recurrence reads the same starting matrix.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

struct Inputs {
    Tensor Q, K, V;
    Tensor a_raw, beta, log_A;
    Tensor S0;          // shared initial state (host FP32)
};

Inputs make_inputs(int L, int num_heads, int d_k, int d_v, uint64_t seed) {
    SplitMix64 rng(seed);
    Inputs in;
    in.Q     = Tensor::mat(L, num_heads * d_k);
    in.K     = Tensor::mat(L, num_heads * d_k);
    in.V     = Tensor::mat(L, num_heads * d_v);
    in.a_raw = Tensor::mat(L, num_heads);
    in.beta  = Tensor::mat(L, num_heads);
    in.log_A = Tensor::mat(num_heads, 1);
    in.S0    = Tensor::mat(num_heads, d_v * d_k);
    fill_random(in.Q,     rng, 0.3f);
    fill_random(in.K,     rng, 0.3f);
    fill_random(in.V,     rng, 0.3f);
    fill_random(in.a_raw, rng);
    fill_random(in.beta,  rng);
    fill_random(in.log_A, rng, 0.5f);
    fill_random(in.S0,    rng, 0.1f);   // small baseline state
    return in;
}

void run_one(int L, int num_heads, int d_k, int d_v, bool use_step,
             uint64_t seed) {
    Inputs in = make_inputs(L, num_heads, d_k, d_v, seed);

    // CPU run — clone S0 so state mutation doesn't bleed into the GPU path.
    Tensor cpu_S = in.S0.clone();
    Tensor cpu_O;
    if (use_step) {
        brotensor::gated_delta_rule_step(in.Q, in.K, in.V, in.a_raw, in.beta,
                                         in.log_A, num_heads, d_k, d_v,
                                         cpu_S, cpu_O);
    } else {
        brotensor::gated_delta_rule_chunked(in.Q, in.K, in.V, in.a_raw, in.beta,
                                            in.log_A, num_heads, d_k, d_v,
                                            cpu_S, cpu_O);
    }

    // GPU run with the same baseline state.
    Tensor gQ     = in.Q.to(gpu_device());
    Tensor gK     = in.K.to(gpu_device());
    Tensor gV     = in.V.to(gpu_device());
    Tensor ga     = in.a_raw.to(gpu_device());
    Tensor gb     = in.beta.to(gpu_device());
    Tensor glogA  = in.log_A.to(gpu_device());
    Tensor gS     = in.S0.to(gpu_device());
    Tensor gO;
    if (use_step) {
        brotensor::gated_delta_rule_step(gQ, gK, gV, ga, gb, glogA,
                                         num_heads, d_k, d_v, gS, gO);
    } else {
        brotensor::gated_delta_rule_chunked(gQ, gK, gV, ga, gb, glogA,
                                            num_heads, d_k, d_v, gS, gO);
    }

    compare_tensors(cpu_O, download_to_host(gO),
                    use_step ? "gdr_step_O" : "gdr_chunked_O",
                    2e-4f, 2e-3f);
    compare_tensors(cpu_S, download_to_host(gS),
                    use_step ? "gdr_step_state" : "gdr_chunked_state",
                    2e-4f, 2e-3f);
}

} // namespace

// chunked (L > 1)
BT_PARITY_TEST(gdr_chunked_1h_8x8_L4)   { run_one(4, 1, 8,  8,  false, 0x7400ull); }
BT_PARITY_TEST(gdr_chunked_2h_16x16_L7) { run_one(7, 2, 16, 16, false, 0x7401ull); }
BT_PARITY_TEST(gdr_chunked_4h_32x32_L3) { run_one(3, 4, 32, 32, false, 0x7402ull); }
BT_PARITY_TEST(gdr_chunked_dk_ne_dv)    { run_one(5, 2, 16, 24, false, 0x7403ull); }

// step (L == 1) — same kernel, different op name
BT_PARITY_TEST(gdr_step_1h_8x8)         { run_one(1, 1, 8,  8,  true,  0x7410ull); }
BT_PARITY_TEST(gdr_step_4h_32x32)       { run_one(1, 4, 32, 32, true,  0x7411ull); }
BT_PARITY_TEST(gdr_step_dk_ne_dv)       { run_one(1, 2, 16, 24, true,  0x7412ull); }

int main() { return run_all("gated_delta_rule cpu/gpu parity"); }
