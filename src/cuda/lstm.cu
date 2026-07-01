// CUDA implementation of the trainable LSTM cell (ops/lstm.h) — a direct port
// of src/cpu/lstm.cpp. Single-layer, single-direction, PyTorch nn.LSTM layout:
// W_ih (4H,I), W_hh (4H,H), gate-row blocks ordered [input, forget, cell,
// output]; gate g uses tanh, the rest sigmoid. FP32-only, matching the CPU
// contract (the cell is exercised in training, not large-batch inference, so
// there is no FP16/BF16 path).
//
// The recurrence is sequential in time, so the host loops over timesteps and
// launches one kernel per step on the current stream; steps serialize on that
// stream (step t reads the hidden/cell state step t-1 wrote). Within a step
// the (B*H) units are independent and run one-thread-per-unit.
//
//   forward  — each thread owns one unit (b, n): gathers W_ih·x + W_hh·h_prev
//              (+biases), applies the gate nonlinearities, writes its slot of
//              the gate cache, cell state, and hidden output. No atomics.
//   backward — each thread owns one unit (b, n) of the current step: forms the
//              gate-preactivation grads, then scatters into the shared
//              parameter grads (dW_ih/dW_hh/db_ih/db_hh — accumulate), dX
//              (overwritten; zeroed up front), and the recurrent carry
//              dh_prev. Cross-(b) and cross-(n) scatter collisions are resolved
//              with atomicAdd. The cell carry dc_prev is per-unit (no coupling
//              across n), so it is written without atomics.
//
// Carry buffers: dc lives in a single (B,H) buffer updated in place; dh uses a
// ping-pong pair (the incoming carry is read while the outgoing carry is
// atomic-accumulated into a freshly zeroed twin, then the two are swapped).
// After the t=0 step the dh buffer holds dL/dh0 and dc holds dL/dc0.

#include <brotensor/tensor.h>
#include <brotensor/detail/dispatch.h>
#include "detail/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace brotensor {

void* cuda_current_stream();

namespace detail::cuda {

namespace {

constexpr int LSTM_BLOCK = 256;

inline int lstm_grid(int n) {
    int blocks = (n + LSTM_BLOCK - 1) / LSTM_BLOCK;
    return blocks < 1 ? 1 : blocks;
}

[[noreturn]] inline void fail(const char* op, const std::string& reason) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + reason);
}

inline void require_fp32(const char* op, const ::brotensor::Tensor& t,
                         const char* name) {
    if (t.dtype != ::brotensor::Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32 (lstm is FP32-only)");
    }
}

// Device pointer of an optional bias/state tensor, or nullptr when absent.
// Validates dtype + shape when present.
const float* opt_ptr(const char* op, const ::brotensor::Tensor* t,
                     const char* name, int rows, int cols) {
    if (!t || t->empty()) return nullptr;
    require_fp32(op, *t, name);
    if (t->rows != rows || t->cols != cols) {
        fail(op, std::string(name) + " has wrong shape");
    }
    return static_cast<const float*>(t->data);
}

__device__ inline float sigmoidf_d(float x) { return 1.0f / (1.0f + __expf(-x)); }

// Warp-aggregated atomics helper for the backward kernel below. Several of
// its accumulation targets (dxrow[j], dh_out_b[m]) are addressed purely by
// this thread's `b` — independent of `n` — so on every loop iteration, every
// lane in the warp that shares this thread's `b` is atomicAdd-ing the exact
// same address; the bias grads (dbih/dbhh) have the mirror pattern keyed by
// `n`. `mask`/`leader` (from __match_any_sync, computed once by the caller)
// identify that lane subgroup for whatever H/B happen to be in play — no
// alignment between warp size and H is assumed. Every lane relays its value
// through shared memory (indexed by absolute thread id, so concurrent warps
// never touch each other's slots) and only the elected leader lane sums the
// subgroup and returns a meaningful result; the caller must gate the actual
// atomicAdd on `lane == leader`.
//
// A shuffle-based tree reduction would be cheaper than the shared-memory
// relay, but __shfl_*_sync requires every lane named in `mask` to execute
// the identical instruction — a solo "leader" cannot use it to pull values
// from the other lanes on its own. Shared memory has no such restriction, at
// the cost of two __syncwarp() fences per call.
__device__ inline float warp_relay_sum(float* relay, float val,
                                       unsigned mask, int leader,
                                       unsigned active_mask) {
    const int lane = threadIdx.x & 31;
    relay[threadIdx.x] = val;
    __syncwarp(active_mask);
    float sum = val;
    if (lane == leader) {
        sum = 0.0f;
        unsigned remaining = mask;
        const int base = threadIdx.x - lane;
        while (remaining) {
            const int src = __ffs(remaining) - 1;
            remaining &= remaining - 1;
            sum += relay[base + src];
        }
    }
    __syncwarp(active_mask);
    return sum;
}

// ───────────────────────────── forward ────────────────────────────────────
//
// One thread per unit (b, n) of timestep t. h_prev / c_prev come from the
// previous step's outputs (or h0/c0 at t==0); those reads are safe because the
// prior step's kernel has already completed on this stream.
__global__ void lstm_fwd_kernel(const float* __restrict__ xp,
                                const float* __restrict__ wih,
                                const float* __restrict__ whh,
                                const float* __restrict__ bih,
                                const float* __restrict__ bhh,
                                const float* __restrict__ h0p,
                                const float* __restrict__ c0p,
                                float* __restrict__ yp,
                                float* __restrict__ gp,
                                float* __restrict__ cp,
                                int t, int B, int I, int H) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int BH = B * H;
    if (idx >= BH) return;
    const int b = idx / H;
    const int n = idx % H;
    const int G = 4 * H;
    const long long row = static_cast<long long>(t) * B + b;

    const float* x = xp + row * I;
    const float* hprev = (t == 0) ? (h0p ? h0p + static_cast<long long>(b) * H : nullptr)
                                  : yp + (static_cast<long long>(t - 1) * B + b) * H;
    const float* cprev = (t == 0) ? (c0p ? c0p + static_cast<long long>(b) * H : nullptr)
                                  : cp + (static_cast<long long>(t - 1) * B + b) * H;

    float zi = bih ? bih[n]         : 0.0f;
    float zf = bih ? bih[H + n]     : 0.0f;
    float zg = bih ? bih[2 * H + n] : 0.0f;
    float zo = bih ? bih[3 * H + n] : 0.0f;
    if (bhh) { zi += bhh[n]; zf += bhh[H + n]; zg += bhh[2 * H + n]; zo += bhh[3 * H + n]; }

    const float* wi = wih + static_cast<long long>(n) * I;
    const float* wf = wih + static_cast<long long>(H + n) * I;
    const float* wg = wih + static_cast<long long>(2 * H + n) * I;
    const float* wo = wih + static_cast<long long>(3 * H + n) * I;
    for (int j = 0; j < I; ++j) {
        const float xj = x[j];
        zi += wi[j] * xj; zf += wf[j] * xj; zg += wg[j] * xj; zo += wo[j] * xj;
    }
    if (hprev) {
        const float* ui = whh + static_cast<long long>(n) * H;
        const float* uf = whh + static_cast<long long>(H + n) * H;
        const float* ug = whh + static_cast<long long>(2 * H + n) * H;
        const float* uo = whh + static_cast<long long>(3 * H + n) * H;
        for (int m = 0; m < H; ++m) {
            const float hm = hprev[m];
            zi += ui[m] * hm; zf += uf[m] * hm; zg += ug[m] * hm; zo += uo[m] * hm;
        }
    }

    const float ig = sigmoidf_d(zi);
    const float fg = sigmoidf_d(zf);
    const float gg = tanhf(zg);
    const float og = sigmoidf_d(zo);
    const float cprev_n = cprev ? cprev[n] : 0.0f;
    const float cn = fg * cprev_n + ig * gg;

    float* grow = gp + row * G;
    grow[n] = ig; grow[H + n] = fg; grow[2 * H + n] = gg; grow[3 * H + n] = og;
    cp[row * H + n] = cn;
    yp[row * H + n] = og * tanhf(cn);
}

// ───────────────────────────── backward ───────────────────────────────────
//
// One thread per unit (b, n) of timestep t. dh_in is the recurrent grad
// arriving from step t+1; dh_out (pre-zeroed twin) accumulates dL/dh_{t-1};
// dc is read for the incoming cell grad and overwritten with dL/dc_{t-1} for
// this unit. Parameter grads and dX are scattered with atomicAdd.
__global__ void lstm_bwd_kernel(const float* __restrict__ xp,
                                const float* __restrict__ yp,
                                const float* __restrict__ gp,
                                const float* __restrict__ cp,
                                const float* __restrict__ dyp,
                                const float* __restrict__ wih,
                                const float* __restrict__ whh,
                                const float* __restrict__ h0p,
                                const float* __restrict__ c0p,
                                const float* __restrict__ dh_in,
                                float* __restrict__ dh_out,
                                float* __restrict__ dc,
                                float* __restrict__ dxp,
                                float* __restrict__ dwih,
                                float* __restrict__ dwhh,
                                float* __restrict__ dbih,
                                float* __restrict__ dbhh,
                                int t, int B, int I, int H) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int BH = B * H;
    if (idx >= BH) return;
    const int b = idx / H;
    const int n = idx % H;
    const int G = 4 * H;
    const long long row = static_cast<long long>(t) * B + b;

    // Warp-aggregated atomics setup (see warp_relay_sum above). `active` is
    // stable for the remainder of the kernel — no further thread returns or
    // data-dependent branches happen before it (the only later branch,
    // `if (hprev)`, depends solely on `t` and the launch-wide h0p pointer, so
    // it is warp-uniform). bmask/bleader group lanes sharing `b` (dxrow /
    // dh_out targets); nmask/nleader group lanes sharing `n` (bias targets).
    const unsigned active = __activemask();
    const int lane = threadIdx.x & 31;
    const unsigned bmask = __match_any_sync(active, b);
    const int bleader = __ffs(bmask) - 1;
    const unsigned nmask = __match_any_sync(active, n);
    const int nleader = __ffs(nmask) - 1;
    __shared__ float s_relay[LSTM_BLOCK];

    const float* grow = gp + row * G;
    const float ig = grow[n], fg = grow[H + n], gg = grow[2 * H + n], og = grow[3 * H + n];
    const float tc = tanhf(cp[row * H + n]);

    const float dh = dyp[row * H + n] + dh_in[static_cast<long long>(b) * H + n];
    const float do_ = dh * tc;
    const float dcv = dh * og * (1.0f - tc * tc) + dc[static_cast<long long>(b) * H + n];

    const float* cprev = (t == 0) ? (c0p ? c0p + static_cast<long long>(b) * H : nullptr)
                                  : cp + (static_cast<long long>(t - 1) * B + b) * H;
    const float cprev_n = cprev ? cprev[n] : 0.0f;
    const float df = dcv * cprev_n;
    const float di = dcv * gg;
    const float dg = dcv * ig;
    const float dcprev = dcv * fg;

    const float dzi = di * ig * (1.0f - ig);
    const float dzf = df * fg * (1.0f - fg);
    const float dzg = dg * (1.0f - gg * gg);
    const float dzo = do_ * og * (1.0f - og);

    // Bias grads key off `n` only: whenever a warp straddles a b-boundary
    // (only possible for H smaller than the warp width) two or more lanes
    // with the same `n` but different `b` land on the identical dbih[n] /
    // dbhh[n] address. nmask/nleader (computed once above) capture exactly
    // that lane subgroup for any H; reduce once and let the leader issue at
    // most 4+4 atomics per warp group instead of one pair per lane.
    if (dbih || dbhh) {
        const float sum_i = warp_relay_sum(s_relay, dzi, nmask, nleader, active);
        const float sum_f = warp_relay_sum(s_relay, dzf, nmask, nleader, active);
        const float sum_g = warp_relay_sum(s_relay, dzg, nmask, nleader, active);
        const float sum_o = warp_relay_sum(s_relay, dzo, nmask, nleader, active);
        if (lane == nleader) {
            if (dbih) { atomicAdd(&dbih[n], sum_i); atomicAdd(&dbih[H + n], sum_f); atomicAdd(&dbih[2 * H + n], sum_g); atomicAdd(&dbih[3 * H + n], sum_o); }
            if (dbhh) { atomicAdd(&dbhh[n], sum_i); atomicAdd(&dbhh[H + n], sum_f); atomicAdd(&dbhh[2 * H + n], sum_g); atomicAdd(&dbhh[3 * H + n], sum_o); }
        }
    }

    // Input path: dX += W_ih^T·dz ;  dW_ih += dz·x^T
    const float* x = xp + row * I;
    float* dxrow = dxp + row * I;
    const float* wi = wih + static_cast<long long>(n) * I;
    const float* wf = wih + static_cast<long long>(H + n) * I;
    const float* wg = wih + static_cast<long long>(2 * H + n) * I;
    const float* wo = wih + static_cast<long long>(3 * H + n) * I;
    float* dwi = dwih + static_cast<long long>(n) * I;
    float* dwf = dwih + static_cast<long long>(H + n) * I;
    float* dwg = dwih + static_cast<long long>(2 * H + n) * I;
    float* dwo = dwih + static_cast<long long>(3 * H + n) * I;
    for (int j = 0; j < I; ++j) {
        // dxrow[j] is keyed by `b` only (row = t*B + b): whenever H is >=
        // the warp width, every lane in a warp shares the same `b` and would
        // otherwise all atomicAdd the same address on every iteration; for
        // smaller H, bmask still finds whatever same-`b` subgroup exists.
        const float dx_contrib = wi[j] * dzi + wf[j] * dzf + wg[j] * dzg + wo[j] * dzo;
        const float dx_sum = warp_relay_sum(s_relay, dx_contrib, bmask, bleader, active);
        if (lane == bleader) atomicAdd(&dxrow[j], dx_sum);
        const float xj = x[j];
        atomicAdd(&dwi[j], dzi * xj); atomicAdd(&dwf[j], dzf * xj);
        atomicAdd(&dwg[j], dzg * xj); atomicAdd(&dwo[j], dzo * xj);
    }

    // Recurrent path: dh_prev += W_hh^T·dz ;  dW_hh += dz·h_prev^T
    const float* hprev = (t == 0) ? (h0p ? h0p + static_cast<long long>(b) * H : nullptr)
                                  : yp + (static_cast<long long>(t - 1) * B + b) * H;
    const float* ui = whh + static_cast<long long>(n) * H;
    const float* uf = whh + static_cast<long long>(H + n) * H;
    const float* ug = whh + static_cast<long long>(2 * H + n) * H;
    const float* uo = whh + static_cast<long long>(3 * H + n) * H;
    float* dui = dwhh + static_cast<long long>(n) * H;
    float* duf = dwhh + static_cast<long long>(H + n) * H;
    float* dug = dwhh + static_cast<long long>(2 * H + n) * H;
    float* duo = dwhh + static_cast<long long>(3 * H + n) * H;
    float* dh_out_b = dh_out + static_cast<long long>(b) * H;
    for (int m = 0; m < H; ++m) {
        // dh_out_b[m] is keyed by `b` only — same collision pattern as dxrow
        // above, reduced through the same bmask/bleader group.
        const float dh_contrib = ui[m] * dzi + uf[m] * dzf + ug[m] * dzg + uo[m] * dzo;
        const float dh_sum = warp_relay_sum(s_relay, dh_contrib, bmask, bleader, active);
        if (lane == bleader) atomicAdd(&dh_out_b[m], dh_sum);
        if (hprev) {
            const float hm = hprev[m];
            atomicAdd(&dui[m], dzi * hm); atomicAdd(&duf[m], dzf * hm);
            atomicAdd(&dug[m], dzg * hm); atomicAdd(&duo[m], dzo * hm);
        }
    }

    // Cell carry is per-unit (no coupling across n) — plain write, no atomic.
    dc[static_cast<long long>(b) * H + n] = dcprev;
}

}  // namespace

void lstm_forward_train(const ::brotensor::Tensor& X, const ::brotensor::Tensor& W_ih,
                        const ::brotensor::Tensor& W_hh,
                        const ::brotensor::Tensor* b_ih, const ::brotensor::Tensor* b_hh,
                        const ::brotensor::Tensor* h0, const ::brotensor::Tensor* c0,
                        int T, int B,
                        ::brotensor::Tensor& Y, ::brotensor::Tensor& gates,
                        ::brotensor::Tensor& C,
                        ::brotensor::Tensor* hT, ::brotensor::Tensor* cT) {
    const char* op = "lstm_forward_train";
    require_fp32(op, X, "X");
    require_fp32(op, W_ih, "W_ih");
    require_fp32(op, W_hh, "W_hh");
    const int H = W_hh.cols;
    const int I = W_ih.cols;
    const int G = 4 * H;
    if (T <= 0 || B <= 0) fail(op, "T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G) fail(op, "W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I) fail(op, "X must be (T*B, I)");

    const float* bih = opt_ptr(op, b_ih, "b_ih", G, 1);
    const float* bhh = opt_ptr(op, b_hh, "b_hh", G, 1);
    const float* h0p = opt_ptr(op, h0, "h0", B, H);
    const float* c0p = opt_ptr(op, c0, "c0", B, H);

    Y.resize(T * B, H, ::brotensor::Dtype::FP32);
    gates.resize(T * B, G, ::brotensor::Dtype::FP32);
    C.resize(T * B, H, ::brotensor::Dtype::FP32);

    const float* xp = static_cast<const float*>(X.data);
    const float* wih = static_cast<const float*>(W_ih.data);
    const float* whh = static_cast<const float*>(W_hh.data);
    float* yp = static_cast<float*>(Y.data);
    float* gp = static_cast<float*>(gates.data);
    float* cp = static_cast<float*>(C.data);

    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());
    const int grid = lstm_grid(B * H);
    for (int t = 0; t < T; ++t) {
        lstm_fwd_kernel<<<grid, LSTM_BLOCK, 0, stream>>>(
            xp, wih, whh, bih, bhh, h0p, c0p, yp, gp, cp, t, B, I, H);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // hT / cT are the last step's hidden / cell rows.
    if (hT) {
        hT->resize(B, H, ::brotensor::Dtype::FP32);
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
            hT->data, yp + static_cast<long long>(T - 1) * B * H,
            static_cast<size_t>(B) * H * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }
    if (cT) {
        cT->resize(B, H, ::brotensor::Dtype::FP32);
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
            cT->data, cp + static_cast<long long>(T - 1) * B * H,
            static_cast<size_t>(B) * H * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }
}

void lstm_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& W_ih,
                   const ::brotensor::Tensor& W_hh,
                   const ::brotensor::Tensor* h0, const ::brotensor::Tensor* c0,
                   const ::brotensor::Tensor& Y, const ::brotensor::Tensor& gates,
                   const ::brotensor::Tensor& C,
                   const ::brotensor::Tensor& dY, int T, int B,
                   ::brotensor::Tensor& dX, ::brotensor::Tensor& dW_ih,
                   ::brotensor::Tensor& dW_hh,
                   ::brotensor::Tensor* db_ih, ::brotensor::Tensor* db_hh,
                   ::brotensor::Tensor* dh0, ::brotensor::Tensor* dc0) {
    const char* op = "lstm_backward";
    require_fp32(op, X, "X");
    require_fp32(op, W_ih, "W_ih");
    require_fp32(op, W_hh, "W_hh");
    require_fp32(op, Y, "Y");
    require_fp32(op, gates, "gates");
    require_fp32(op, C, "C");
    require_fp32(op, dY, "dY");
    const int H = W_hh.cols;
    const int I = W_ih.cols;
    const int G = 4 * H;
    if (T <= 0 || B <= 0) fail(op, "T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G) fail(op, "W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I) fail(op, "X must be (T*B, I)");
    if (dY.rows != T * B || dY.cols != H) fail(op, "dY must be (T*B, H)");

    // Parameter grads accumulate — require pre-sized buffers, do NOT zero them.
    require_fp32(op, dW_ih, "dW_ih");
    require_fp32(op, dW_hh, "dW_hh");
    if (dW_ih.rows != G || dW_ih.cols != I) fail(op, "dW_ih must be (4H, I), zeroed");
    if (dW_hh.rows != G || dW_hh.cols != H) fail(op, "dW_hh must be (4H, H), zeroed");
    if (db_ih) { require_fp32(op, *db_ih, "db_ih"); if (db_ih->rows != G || db_ih->cols != 1) fail(op, "db_ih must be (4H, 1), zeroed"); }
    if (db_hh) { require_fp32(op, *db_hh, "db_hh"); if (db_hh->rows != G || db_hh->cols != 1) fail(op, "db_hh must be (4H, 1), zeroed"); }

    const float* h0p = opt_ptr(op, h0, "h0", B, H);
    const float* c0p = opt_ptr(op, c0, "c0", B, H);

    dX.resize(T * B, I, ::brotensor::Dtype::FP32);

    const float* xp  = static_cast<const float*>(X.data);
    const float* yp  = static_cast<const float*>(Y.data);
    const float* gp  = static_cast<const float*>(gates.data);
    const float* cp  = static_cast<const float*>(C.data);
    const float* dyp = static_cast<const float*>(dY.data);
    const float* wih = static_cast<const float*>(W_ih.data);
    const float* whh = static_cast<const float*>(W_hh.data);
    float* dxp  = static_cast<float*>(dX.data);
    float* dwih = static_cast<float*>(dW_ih.data);
    float* dwhh = static_cast<float*>(dW_hh.data);
    float* dbih = db_ih ? static_cast<float*>(db_ih->data) : nullptr;
    float* dbhh = db_hh ? static_cast<float*>(db_hh->data) : nullptr;

    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream());

    // dX is overwritten — zero it before the accumulating scatter.
    BROTENSOR_CUDA_CHECK(cudaMemsetAsync(
        dxp, 0, static_cast<size_t>(T) * B * I * sizeof(float), stream));

    // Recurrent carry: dc updated in place; dh ping-pongs between two twins.
    ::brotensor::Tensor dhA = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, B, H, ::brotensor::Dtype::FP32);
    ::brotensor::Tensor dhB = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, B, H, ::brotensor::Dtype::FP32);
    ::brotensor::Tensor dcBuf = ::brotensor::Tensor::zeros_on(::brotensor::Device::CUDA, B, H, ::brotensor::Dtype::FP32);
    float* dh_in  = static_cast<float*>(dhA.data);
    float* dh_out = static_cast<float*>(dhB.data);
    float* dc     = static_cast<float*>(dcBuf.data);
    const size_t bh_bytes = static_cast<size_t>(B) * H * sizeof(float);

    const int grid = lstm_grid(B * H);
    for (int t = T - 1; t >= 0; --t) {
        // Fresh accumulator for dL/dh_{t-1}.
        BROTENSOR_CUDA_CHECK(cudaMemsetAsync(dh_out, 0, bh_bytes, stream));
        lstm_bwd_kernel<<<grid, LSTM_BLOCK, 0, stream>>>(
            xp, yp, gp, cp, dyp, wih, whh, h0p, c0p,
            dh_in, dh_out, dc, dxp, dwih, dwhh, dbih, dbhh,
            t, B, I, H);
        std::swap(dh_in, dh_out);
    }
    BROTENSOR_CUDA_CHECK(cudaGetLastError());

    // After the t=0 step, dh_in holds dL/dh0 and dc holds dL/dc0.
    if (dh0) {
        dh0->resize(B, H, ::brotensor::Dtype::FP32);
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
            dh0->data, dh_in, bh_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    if (dc0) {
        dc0->resize(B, H, ::brotensor::Dtype::FP32);
        BROTENSOR_CUDA_CHECK(cudaMemcpyAsync(
            dc0->data, dc, bh_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    // dhA/dhB/dcBuf free here (Tensor destructors -> cuda_free). That routes
    // through cudaFreeAsync on this same stream when the pooled allocator is
    // active (tensor.cu), which is itself stream-ordered — it queues behind
    // every kernel/copy launched above without needing the host to wait, so
    // freeing them does not require a synchronize. On the non-pooled fallback
    // path cuda_free() calls the synchronous cudaFree(), which implicitly
    // waits out the device itself; either way no explicit host sync is
    // needed here, and dropping it lets this backward pass pipeline with
    // whatever the caller enqueues next instead of stalling the host.
}

void fill_cuda_vtable_lstm(::brotensor::detail::OpsVTable& v) {
    v.lstm_forward_train = &lstm_forward_train;
    v.lstm_backward      = &lstm_backward;
}

}  // namespace detail::cuda
}  // namespace brotensor
