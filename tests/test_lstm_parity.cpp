// CPU↔GPU parity for the trainable LSTM cell (ops/lstm.h).
//
//   lstm_forward_train — Y / gates / C OVERWRITTEN (resized); hT / cT are the
//                        last step's hidden / cell rows.
//   lstm_backward      — dX / dh0 / dc0 OVERWRITTEN; dW_ih / dW_hh / db_ih /
//                        db_hh ACCUMULATE (+=). The accumulation tests pre-fill
//                        the param grads with a non-zero baseline to verify the
//                        contract holds identically on both backends.
//
// PyTorch nn.LSTM layout: W_ih (4H,I), W_hh (4H,H), gate-row order
// [input, forget, cell, output]. Rows of X / Y / gates / C are indexed
// t*B + b. Covered with and without biases and initial states (h0/c0), since
// those are optional-pointer paths that branch in the kernel. FP32-only on
// every backend.

#include "parity_helpers.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

using namespace bt_parity;
using brotensor::Tensor;

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-3f;

struct LstmCfg {
    int T, B, I, H;
    bool bias;    // b_ih + b_hh present
    bool state;   // h0 + c0 present
};

// ─── forward: Y / gates / C (+ hT / cT) ────────────────────────────────────
void run_forward(const LstmCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int G = 4 * c.H;
    Tensor X   = Tensor::mat(c.T * c.B, c.I);
    Tensor Wih = Tensor::mat(G, c.I);
    Tensor Whh = Tensor::mat(G, c.H);
    fill_random(X, rng, 0.6f);
    fill_random(Wih, rng, 0.5f);
    fill_random(Whh, rng, 0.5f);
    Tensor bih, bhh, h0, c0;
    Tensor *bihp = nullptr, *bhhp = nullptr, *h0p = nullptr, *c0p = nullptr;
    if (c.bias) {
        bih = Tensor::mat(G, 1); bhh = Tensor::mat(G, 1);
        fill_random(bih, rng, 0.3f); fill_random(bhh, rng, 0.3f);
        bihp = &bih; bhhp = &bhh;
    }
    if (c.state) {
        h0 = Tensor::mat(c.B, c.H); c0 = Tensor::mat(c.B, c.H);
        fill_random(h0, rng, 0.4f); fill_random(c0, rng, 0.4f);
        h0p = &h0; c0p = &c0;
    }

    Tensor cY, cG, cC, chT, ccT;
    brotensor::lstm_forward_train(X, Wih, Whh, bihp, bhhp, h0p, c0p, c.T, c.B,
                                  cY, cG, cC, &chT, &ccT);

    Tensor gX = X.to(gpu_device()), gWih = Wih.to(gpu_device()), gWhh = Whh.to(gpu_device());
    Tensor gbih, gbhh, gh0, gc0;
    Tensor *gbihp = nullptr, *gbhhp = nullptr, *gh0p = nullptr, *gc0p = nullptr;
    if (c.bias) { gbih = bih.to(gpu_device()); gbhh = bhh.to(gpu_device()); gbihp = &gbih; gbhhp = &gbhh; }
    if (c.state) { gh0 = h0.to(gpu_device()); gc0 = c0.to(gpu_device()); gh0p = &gh0; gc0p = &gc0; }
    Tensor gY, gG, gC, ghT, gcT;
    brotensor::lstm_forward_train(gX, gWih, gWhh, gbihp, gbhhp, gh0p, gc0p, c.T, c.B,
                                  gY, gG, gC, &ghT, &gcT);

    compare_tensors(cY, download_to_host(gY), "lstm_fwd_Y", kAtol, kRtol);
    compare_tensors(cG, download_to_host(gG), "lstm_fwd_gates", kAtol, kRtol);
    compare_tensors(cC, download_to_host(gC), "lstm_fwd_C", kAtol, kRtol);
    compare_tensors(chT, download_to_host(ghT), "lstm_fwd_hT", kAtol, kRtol);
    compare_tensors(ccT, download_to_host(gcT), "lstm_fwd_cT", kAtol, kRtol);
}

// ─── backward: dX / dW_ih / dW_hh / db_ih / db_hh / dh0 / dc0 ───────────────
void run_backward(const LstmCfg& c, uint64_t seed) {
    SplitMix64 rng(seed);
    const int G = 4 * c.H;
    Tensor X   = Tensor::mat(c.T * c.B, c.I);
    Tensor Wih = Tensor::mat(G, c.I);
    Tensor Whh = Tensor::mat(G, c.H);
    fill_random(X, rng, 0.6f);
    fill_random(Wih, rng, 0.5f);
    fill_random(Whh, rng, 0.5f);
    Tensor bih, bhh, h0, c0;
    Tensor *bihp = nullptr, *bhhp = nullptr, *h0p = nullptr, *c0p = nullptr;
    if (c.bias) {
        bih = Tensor::mat(G, 1); bhh = Tensor::mat(G, 1);
        fill_random(bih, rng, 0.3f); fill_random(bhh, rng, 0.3f);
        bihp = &bih; bhhp = &bhh;
    }
    if (c.state) {
        h0 = Tensor::mat(c.B, c.H); c0 = Tensor::mat(c.B, c.H);
        fill_random(h0, rng, 0.4f); fill_random(c0, rng, 0.4f);
        h0p = &h0; c0p = &c0;
    }

    // Forward on the CPU to produce the caches the backward consumes, plus an
    // upstream grad dY.
    Tensor cY, cG, cC;
    brotensor::lstm_forward_train(X, Wih, Whh, bihp, bhhp, h0p, c0p, c.T, c.B,
                                  cY, cG, cC, nullptr, nullptr);
    Tensor dY = Tensor::mat(c.T * c.B, c.H);
    fill_random(dY, rng, 0.5f);

    // Non-zero baselines for the accumulating param grads — identical on both
    // backends so the += contribution is what's compared.
    Tensor dWih0 = Tensor::mat(G, c.I); fill_random(dWih0, rng, 0.2f);
    Tensor dWhh0 = Tensor::mat(G, c.H); fill_random(dWhh0, rng, 0.2f);
    Tensor dbih0 = Tensor::mat(G, 1);   fill_random(dbih0, rng, 0.2f);
    Tensor dbhh0 = Tensor::mat(G, 1);   fill_random(dbhh0, rng, 0.2f);

    // ── CPU ──
    Tensor cdX, cdWih = dWih0, cdWhh = dWhh0, cdbih = dbih0, cdbhh = dbhh0;
    Tensor cdh0, cdc0;
    brotensor::lstm_backward(X, Wih, Whh, h0p, c0p, cY, cG, cC, dY, c.T, c.B,
                             cdX, cdWih, cdWhh, &cdbih, &cdbhh, &cdh0, &cdc0);

    // ── GPU ──
    Tensor gX = X.to(gpu_device()), gWih = Wih.to(gpu_device()), gWhh = Whh.to(gpu_device());
    Tensor gh0, gc0;
    Tensor *gh0p = nullptr, *gc0p = nullptr;
    if (c.state) { gh0 = h0.to(gpu_device()); gc0 = c0.to(gpu_device()); gh0p = &gh0; gc0p = &gc0; }
    Tensor gY = cY.to(gpu_device()), gG = cG.to(gpu_device()), gC = cC.to(gpu_device());
    Tensor gdY = dY.to(gpu_device());
    Tensor gdX, gdWih = dWih0.to(gpu_device()), gdWhh = dWhh0.to(gpu_device());
    Tensor gdbih = dbih0.to(gpu_device()), gdbhh = dbhh0.to(gpu_device());
    Tensor gdh0, gdc0;
    brotensor::lstm_backward(gX, gWih, gWhh, gh0p, gc0p, gY, gG, gC, gdY, c.T, c.B,
                             gdX, gdWih, gdWhh, &gdbih, &gdbhh, &gdh0, &gdc0);

    compare_tensors(cdX,   download_to_host(gdX),   "lstm_bwd_dX",   kAtol, kRtol);
    compare_tensors(cdWih, download_to_host(gdWih), "lstm_bwd_dWih", kAtol, kRtol);
    compare_tensors(cdWhh, download_to_host(gdWhh), "lstm_bwd_dWhh", kAtol, kRtol);
    compare_tensors(cdbih, download_to_host(gdbih), "lstm_bwd_dbih", kAtol, kRtol);
    compare_tensors(cdbhh, download_to_host(gdbhh), "lstm_bwd_dbhh", kAtol, kRtol);
    compare_tensors(cdh0,  download_to_host(gdh0),  "lstm_bwd_dh0",  kAtol, kRtol);
    compare_tensors(cdc0,  download_to_host(gdc0),  "lstm_bwd_dc0",  kAtol, kRtol);
}

const LstmCfg kSmall    {4, 2, 3, 5,  true,  true};
const LstmCfg kNoBias   {4, 2, 3, 5,  false, true};
const LstmCfg kNoState  {4, 2, 3, 5,  true,  false};
const LstmCfg kBare     {3, 1, 4, 4,  false, false};
const LstmCfg kWide     {5, 3, 8, 7,  true,  true};
const LstmCfg kSingleT  {1, 2, 3, 5,  true,  true};

} // namespace

BT_PARITY_TEST(lstm_fwd_small)    { run_forward(kSmall,   0x1a00ull); }
BT_PARITY_TEST(lstm_fwd_nobias)   { run_forward(kNoBias,  0x1a01ull); }
BT_PARITY_TEST(lstm_fwd_nostate)  { run_forward(kNoState, 0x1a02ull); }
BT_PARITY_TEST(lstm_fwd_bare)     { run_forward(kBare,    0x1a03ull); }
BT_PARITY_TEST(lstm_fwd_wide)     { run_forward(kWide,    0x1a04ull); }
BT_PARITY_TEST(lstm_fwd_singlet)  { run_forward(kSingleT, 0x1a05ull); }

BT_PARITY_TEST(lstm_bwd_small)    { run_backward(kSmall,   0x1b00ull); }
BT_PARITY_TEST(lstm_bwd_nobias)   { run_backward(kNoBias,  0x1b01ull); }
BT_PARITY_TEST(lstm_bwd_nostate)  { run_backward(kNoState, 0x1b02ull); }
BT_PARITY_TEST(lstm_bwd_bare)     { run_backward(kBare,    0x1b03ull); }
BT_PARITY_TEST(lstm_bwd_wide)     { run_backward(kWide,    0x1b04ull); }
BT_PARITY_TEST(lstm_bwd_singlet)  { run_backward(kSingleT, 0x1b05ull); }

int main() { return run_all("lstm cpu/gpu parity"); }
