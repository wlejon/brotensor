// CPU↔GPU parity tests for the NCHW <-> sequence transposes (CHUNK 4).
//
//   nchw_to_sequence : (N, C*H*W) -> (N*HW, C),  Y[(n*HW+p)*C + c]
//   sequence_to_nchw : (N*HW, C)  -> (N, C*H*W), exact inverse
//
// Both ops are pure gathers — no arithmetic, no rounding — so FP32 parity is
// exact (atol/rtol both 0). Also exercises the round-trip identity.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;
using brotensor::Device;

namespace {

struct Shape { int N, C, H, W; };

void run_fwd(const Shape& s, uint64_t seed) {
    SplitMix64 rng(seed);
    const int HW = s.H * s.W;
    Tensor X = Tensor::mat(s.N, s.C * HW);
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::nchw_to_sequence(X, s.N, s.C, s.H, s.W, cpu_Y);

    Tensor gX = X.to(Device::CUDA);
    Tensor gpu_Y;
    brotensor::nchw_to_sequence(gX, s.N, s.C, s.H, s.W, gpu_Y);

    // Pure gather — bit-exact.
    compare_tensors(cpu_Y, download_to_host(gpu_Y), "nchw_to_seq", 0.0f, 0.0f);
}

void run_inv(const Shape& s, uint64_t seed) {
    SplitMix64 rng(seed);
    const int HW = s.H * s.W;
    Tensor X = Tensor::mat(s.N * HW, s.C);   // sequence-layout input
    fill_random(X, rng);

    Tensor cpu_Y;
    brotensor::sequence_to_nchw(X, s.N, s.C, s.H, s.W, cpu_Y);

    Tensor gX = X.to(Device::CUDA);
    Tensor gpu_Y;
    brotensor::sequence_to_nchw(gX, s.N, s.C, s.H, s.W, gpu_Y);

    compare_tensors(cpu_Y, download_to_host(gpu_Y), "seq_to_nchw", 0.0f, 0.0f);
}

// Round-trip on the CPU: sequence_to_nchw(nchw_to_sequence(X)) == X.
void run_roundtrip(const Shape& s, uint64_t seed) {
    SplitMix64 rng(seed);
    const int HW = s.H * s.W;
    Tensor X = Tensor::mat(s.N, s.C * HW);
    fill_random(X, rng);

    Tensor seq, back;
    brotensor::nchw_to_sequence(X, s.N, s.C, s.H, s.W, seq);
    brotensor::sequence_to_nchw(seq, s.N, s.C, s.H, s.W, back);

    compare_tensors(X, back, "roundtrip", 0.0f, 0.0f);
}

const Shape kTiny   {1, 4, 2, 3};
const Shape kSquare {1, 8, 4, 4};
const Shape kBatch  {2, 6, 3, 5};
const Shape kBig    {1, 32, 8, 8};

} // namespace

BT_PARITY_TEST(transpose_nchw_to_seq_tiny)   { run_fwd(kTiny,   0x7100ull); }
BT_PARITY_TEST(transpose_nchw_to_seq_square) { run_fwd(kSquare, 0x7101ull); }
BT_PARITY_TEST(transpose_nchw_to_seq_batch)  { run_fwd(kBatch,  0x7102ull); }
BT_PARITY_TEST(transpose_nchw_to_seq_big)    { run_fwd(kBig,    0x7103ull); }

BT_PARITY_TEST(transpose_seq_to_nchw_tiny)   { run_inv(kTiny,   0x7110ull); }
BT_PARITY_TEST(transpose_seq_to_nchw_square) { run_inv(kSquare, 0x7111ull); }
BT_PARITY_TEST(transpose_seq_to_nchw_batch)  { run_inv(kBatch,  0x7112ull); }
BT_PARITY_TEST(transpose_seq_to_nchw_big)    { run_inv(kBig,    0x7113ull); }

BT_PARITY_TEST(transpose_roundtrip_tiny)   { run_roundtrip(kTiny,   0x7120ull); }
BT_PARITY_TEST(transpose_roundtrip_batch)  { run_roundtrip(kBatch,  0x7121ull); }
BT_PARITY_TEST(transpose_roundtrip_big)    { run_roundtrip(kBig,    0x7122ull); }

int main() { return run_all("transpose cpu/gpu parity"); }
