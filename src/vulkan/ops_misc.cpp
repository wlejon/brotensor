// Vulkan chunk-6 small ops (shaders/misc.comp): the embedding lookup (a row
// gather), masked mean pooling and the slot / causal masks (brogameagent,
// LLM2Vec), threshold_u8 and rows_count_above (SAM's mask post-processing),
// xavier_init, the SGD / Adam steps and the MSE losses, attention_token_moments,
// and StyleGAN3's bias_act and upfirdn2d (filtered_lrelu has no slot here: the
// public op falls back to its bias_act + upfirdn2d composite).
// Contracts follow the CUDA backend (src/cuda/embedding.cu, reduce.cu,
// elementwise.cu, public_reductions.cu, xavier_init.cu, optim.cu, loss.cu,
// attention_moments.cu, bias_act.cu, upfirdn2d.cu): the pooling, bias_act and
// upfirdn2d take FP32 / FP16 / BF16 (FP32 arithmetic); masks, optimiser
// state, losses and moments are FP32; threshold_u8 / rows_count_above read
// FP32 / FP16 (BF16 too here). Masks are valid where >= 0.5 (the CPU's rule).

#include "detail/kernels.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

void gather_rows(const Tensor& X, const Tensor& Idx, Tensor& Y);   // ops_spatial.cpp

namespace {

struct MiscPush {
    std::uint64_t a, b, c, d, e;
    std::uint32_t u[16];
    float f[4];
};

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void ensure(Tensor& t, long long r, long long c, Dtype dt, const char* op) {
    if (r < 0 || c < 0 || r * c > 0x7fffffffLL) fail(op, "output too large");
    if (t.rows != r || t.cols != c || t.dtype != dt) t.resize(static_cast<int>(r), static_cast<int>(c), dt);
}

std::uint32_t u32(long long v) { return static_cast<std::uint32_t>(v); }

const Kernel& misc_kernel(DeviceCtx& d, Dtype dt, std::uint32_t code, const char* op) {
    return d.pipelines().get(dt_variant(ShaderId::misc_f32, dt, op), {code});
}

void run_items(DeviceCtx& d, Dtype dt, std::uint32_t code, const MiscPush& pc, std::uint64_t items, const char* op) {
    if (items == 0) return;
    const Kernel& k = misc_kernel(d, dt, code, op);
    launch(d, k, pc, groups_1d(items, k));
}

void run_groups(DeviceCtx& d, Dtype dt, std::uint32_t code, const MiscPush& pc, std::uint64_t groups, const char* op) {
    if (groups == 0) return;
    const Kernel& k = misc_kernel(d, dt, code, op);
    const std::uint32_t gx = static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, 65535));
    const std::uint32_t gy = static_cast<std::uint32_t>(std::min<std::uint64_t>((groups + gx - 1) / gx, 65535));
    launch(d, k, pc, gx, gy);
}

void need_f32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(name) + " must be FP32");
}

}  // namespace

// ─── embedding, pooling, masks ─────────────────────────────────────────────

// A row gather of the table by the device-resident INT32 indices (an index
// outside the table is clamped rather than faulting the device).
void embedding_lookup_forward(const Tensor& table, const int32_t* d_idx, int B, Tensor& out) {
    const char* op = "embedding_lookup_forward";
    dt_code(table.dtype, op);
    need(op, B >= 0, "B must be >= 0");
    ensure(out, B, table.cols, table.dtype, op);
    if (B == 0 || table.cols == 0) return;
    need(op, d_idx != nullptr, "null index pointer");
    const Tensor idx = Tensor::view(table.device, const_cast<int32_t*>(d_idx), B, 1, Dtype::INT32);
    gather_rows(table, idx, out);
}

void masked_mean_pool_forward(const Tensor& X, const float* d_mask, Tensor& y) {
    const char* op = "masked_mean_pool_forward";
    dt_code(X.dtype, op);
    ensure(y, X.cols, 1, X.dtype, op);
    if (X.cols == 0) return;
    MiscPush pc{};
    pc.a = addr(X.data); pc.b = addr(d_mask); pc.c = addr(y.data);
    pc.u[0] = u32(X.rows); pc.u[1] = u32(X.cols);
    run_items(device_of(X), X.dtype, MI_MEAN_POOL, pc, u32(X.cols), op);
}

void masked_mean_pool_backward(const Tensor& dY, const float* d_mask, int K, Tensor& dX) {
    const char* op = "masked_mean_pool_backward";
    dt_code(dY.dtype, op);
    need(op, K >= 0, "K must be >= 0");
    const long long D = dY.size();
    ensure(dX, K, D, dY.dtype, op);
    MiscPush pc{};
    pc.a = addr(dY.data); pc.b = addr(d_mask); pc.c = addr(dX.data);
    pc.u[0] = u32(K); pc.u[1] = u32(D);
    run_items(device_of(dY), dY.dtype, MI_MEAN_POOL_BWD, pc, std::uint64_t(K) * u32(D), op);
}

void build_slot_mask(const Tensor& x, int offset, int K, int stride, Tensor& mask) {
    const char* op = "build_slot_mask";
    need_f32(op, x, "x");
    need(op, K >= 0 && offset >= 0 && stride >= 0, "offset, K and stride must be >= 0");
    need(op, K == 0 || offset + static_cast<long long>(K - 1) * stride < x.size(), "slots reach past x");
    ensure(mask, K, 1, Dtype::FP32, op);
    MiscPush pc{};
    pc.a = addr(x.data); pc.c = addr(mask.data);
    pc.u[0] = u32(K); pc.u[1] = u32(offset); pc.u[2] = u32(stride);
    run_items(device_of(x), Dtype::FP32, MI_SLOT_MASK, pc, u32(K), op);
}

void build_causal_mask_row(int L, int q, Tensor& mask) {
    const char* op = "build_causal_mask_row";
    ensure(mask, std::max(L, 0), 1, Dtype::FP32, op);
    if (L <= 0) return;
    MiscPush pc{};
    pc.c = addr(mask.data);
    pc.u[0] = u32(L); pc.u[1] = static_cast<std::uint32_t>(q);
    run_items(device_of(mask), Dtype::FP32, MI_CAUSAL_ROW, pc, u32(L), op);
}

void threshold_u8(const Tensor& X, float t, Tensor& Y) {
    const char* op = "threshold_u8";
    dt_code(X.dtype, op);
    ensure(Y, X.rows, X.cols, Dtype::INT8, op);
    MiscPush pc{};
    pc.a = addr(X.data); pc.c = addr(Y.data);
    pc.u[0] = count32(X, op);
    pc.f[0] = t;
    run_items(device_of(X), X.dtype, MI_THRESHOLD_U8, pc, pc.u[0], op);
}

void rows_count_above(const Tensor& X, float t_lo, float t_hi, Tensor& counts) {
    const char* op = "rows_count_above";
    dt_code(X.dtype, op);
    ensure(counts, X.rows, 2, Dtype::INT32, op);
    if (X.rows == 0) return;
    if (X.cols == 0) { counts.zero(); return; }
    MiscPush pc{};
    pc.a = addr(X.data); pc.c = addr(counts.data);
    pc.u[0] = u32(X.rows); pc.u[1] = u32(X.cols);
    pc.f[0] = t_lo; pc.f[1] = t_hi;
    run_groups(device_of(X), X.dtype, MI_COUNT_ABOVE, pc, u32(X.rows), op);
}

// ─── init, optimisers, losses ──────────────────────────────────────────────

// Bit-identical to the CPU's sequential splitmix64 walk: element i draws at
// state + (i + 1) K, and the state advances by n K.
void xavier_init(Tensor& W, uint64_t& rng_state) {
    const char* op = "xavier_init";
    need_f32(op, W, "W");
    const std::uint32_t n = count32(W, op);
    if (n == 0) return;
    const float limit = std::sqrt(6.0f / static_cast<float>(W.rows + W.cols));
    MiscPush pc{};
    pc.c = addr(W.data);
    pc.u[0] = n;
    pc.u[1] = static_cast<std::uint32_t>(rng_state);
    pc.u[2] = static_cast<std::uint32_t>(rng_state >> 32);
    pc.f[0] = limit;
    run_items(device_of(W), Dtype::FP32, MI_XAVIER, pc, n, op);
    rng_state += static_cast<std::uint64_t>(n) * 0x9E3779B97F4A7C15ULL;
}

void sgd_step(Tensor& param, Tensor& grad, Tensor& velocity, float lr, float momentum) {
    const char* op = "sgd_step";
    for (const Tensor* t : std::initializer_list<const Tensor*>{&param, &grad, &velocity}) need_f32(op, *t, "param, grad and velocity");
    need(op, grad.size() == param.size() && velocity.size() == param.size(), "size mismatch");
    MiscPush pc{};
    pc.a = addr(param.data); pc.b = addr(grad.data); pc.c = addr(velocity.data);
    pc.u[0] = count32(param, op);
    pc.f[0] = lr; pc.f[1] = momentum;
    run_items(device_of(param), Dtype::FP32, MI_SGD, pc, pc.u[0], op);
}

void adam_step(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v, float lr, float beta1, float beta2, float eps,
               int step) {
    const char* op = "adam_step";
    for (const Tensor* t : std::initializer_list<const Tensor*>{&param, &grad, &m, &v}) need_f32(op, *t, "param, grad, m and v");
    need(op, grad.size() == param.size() && m.size() == param.size() && v.size() == param.size(), "size mismatch");
    const float ibc1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
    const float ibc2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));
    MiscPush pc{};
    pc.a = addr(param.data); pc.b = addr(grad.data); pc.c = addr(m.data); pc.d = addr(v.data);
    pc.u[0] = count32(param, op);
    std::memcpy(&pc.u[1], &ibc1, 4);
    std::memcpy(&pc.u[2], &ibc2, 4);
    pc.f[0] = lr; pc.f[1] = beta1; pc.f[2] = beta2; pc.f[3] = eps;
    run_items(device_of(param), Dtype::FP32, MI_ADAM, pc, pc.u[0], op);
}

float mse_vec_forward(const Tensor& pred, const Tensor& target) {
    const char* op = "mse_vec_forward";
    need_f32(op, pred, "pred");
    need_f32(op, target, "target");
    need(op, target.size() == pred.size(), "size mismatch");
    const std::uint32_t n = count32(pred, op);
    if (n == 0) return 0.0f;
    Tensor out = Tensor::empty_on(pred.device, 1, 1, Dtype::FP32);
    MiscPush pc{};
    pc.a = addr(pred.data); pc.b = addr(target.data); pc.c = addr(out.data);
    pc.u[0] = n;
    DeviceCtx& d = device_of(pred);
    launch(d, misc_kernel(d, Dtype::FP32, MI_MSE_SUM, op), pc, 1);
    float s = 0.0f;
    out.copy_to_host_raw(&s, sizeof(float));
    return s / static_cast<float>(n);
}

void mse_vec_backward(const Tensor& pred, const Tensor& target, Tensor& dPred) {
    const char* op = "mse_vec_backward";
    need_f32(op, pred, "pred");
    need_f32(op, target, "target");
    need(op, target.size() == pred.size(), "size mismatch");
    ensure(dPred, pred.rows, pred.cols, Dtype::FP32, op);
    const std::uint32_t n = count32(pred, op);
    MiscPush pc{};
    pc.a = addr(pred.data); pc.b = addr(target.data); pc.c = addr(dPred.data);
    pc.u[0] = n;
    pc.f[0] = n ? 2.0f / static_cast<float>(n) : 0.0f;
    run_items(device_of(pred), Dtype::FP32, MI_MSE_BWD, pc, n, op);
}

void mse_vec_per_sample(const Tensor& pred, const Tensor& target, Tensor& dPred, Tensor& loss_per_sample) {
    const char* op = "mse_vec_per_sample";
    need_f32(op, pred, "pred");
    need_f32(op, target, "target");
    need(op, target.size() == pred.size(), "size mismatch");
    ensure(dPred, pred.rows, pred.cols, Dtype::FP32, op);
    ensure(loss_per_sample, pred.size(), 1, Dtype::FP32, op);
    MiscPush pc{};
    pc.a = addr(pred.data); pc.b = addr(target.data); pc.c = addr(dPred.data); pc.d = addr(loss_per_sample.data);
    pc.u[0] = count32(pred, op);
    run_items(device_of(pred), Dtype::FP32, MI_MSE_SAMPLE, pc, pc.u[0], op);
}

// ─── cross-attention moments ───────────────────────────────────────────────

void attention_token_moments(const Tensor& Attn, int h_lat, int w_lat, Tensor& mass, Tensor& centroid) {
    const char* op = "attention_token_moments";
    dt_code(Attn.dtype, op);
    need(op, h_lat > 0 && w_lat > 0, "h_lat and w_lat must be positive");
    need(op, Attn.rows == h_lat * w_lat, "Attn.rows must equal h_lat * w_lat");
    const int Lk = Attn.cols;
    ensure(mass, Lk, 1, Dtype::FP32, op);
    ensure(centroid, Lk, 2, Dtype::FP32, op);
    MiscPush pc{};
    pc.a = addr(Attn.data); pc.c = addr(mass.data); pc.d = addr(centroid.data);
    pc.u[0] = u32(Attn.rows); pc.u[1] = u32(Lk); pc.u[2] = u32(w_lat);
    run_groups(device_of(Attn), Attn.dtype, MI_MOMENTS, pc, u32(Lk), op);
}

// ─── StyleGAN3 bias_act / upfirdn2d ────────────────────────────────────────

namespace {

void check_bias_act(const char* op, const Tensor& X, const Tensor* b, int N, int C, int HW, int act) {
    dt_code(X.dtype, op);
    need(op, act == 0 || act == 1, "act must be 0 (linear) or 1 (lrelu)");
    need(op, N >= 0 && C >= 0 && HW >= 0, "negative dimension");
    need(op, X.rows == N && X.cols == static_cast<long long>(C) * HW, "X shape mismatch");
    if (b && b->data) {
        need(op, b->dtype == X.dtype, "b.dtype must match X.dtype");
        need(op, b->size() == C, "b must have C elements");
    }
}

std::uint64_t opt_addr(const Tensor* t) { return t && t->data ? addr(t->data) : 0; }

}  // namespace

void bias_act_forward(const Tensor& X, const Tensor* b, int N, int C, int HW, int act, float alpha, float gain,
                      float clamp, Tensor& Y) {
    const char* op = "bias_act_forward";
    check_bias_act(op, X, b, N, C, HW, act);
    ensure(Y, N, static_cast<long long>(C) * HW, X.dtype, op);
    MiscPush pc{};
    pc.a = addr(X.data); pc.b = opt_addr(b); pc.c = addr(Y.data);
    pc.u[0] = count32(X, op); pc.u[1] = u32(C); pc.u[2] = u32(std::max(HW, 1)); pc.u[3] = u32(act);
    pc.f[0] = alpha; pc.f[1] = gain; pc.f[2] = clamp;
    run_items(device_of(X), X.dtype, MI_BIAS_ACT, pc, pc.u[0], op);
}

void bias_act_backward(const Tensor& dY, const Tensor& X, const Tensor* b, int N, int C, int HW, int act, float alpha,
                       float gain, float clamp, Tensor& dX, Tensor* dB) {
    const char* op = "bias_act_backward";
    check_bias_act(op, X, b, N, C, HW, act);
    need(op, dY.dtype == X.dtype, "dY.dtype must match X.dtype");
    need(op, dY.rows == N && dY.cols == X.cols, "dY shape mismatch");
    if (dB) {
        need(op, dB->dtype == X.dtype, "dB.dtype must match X.dtype");
        need(op, dB->size() == C, "dB must have C elements");
    }
    ensure(dX, N, X.cols, X.dtype, op);
    if (N == 0 || X.cols == 0) return;
    DeviceCtx& d = device_of(X);
    MiscPush pc{};
    pc.a = addr(dY.data); pc.b = addr(X.data); pc.c = opt_addr(b); pc.d = addr(dX.data);
    pc.u[0] = count32(X, op); pc.u[1] = u32(C); pc.u[2] = u32(HW); pc.u[3] = u32(act);
    pc.f[0] = alpha; pc.f[1] = gain; pc.f[2] = clamp;
    run_items(d, X.dtype, MI_BIAS_ACT_BWD, pc, pc.u[0], op);
    if (dB && dB->data) {   // dB[c] += sum over (n, k) of the gradient
        pc.e = addr(dB->data);
        pc.u[0] = u32(N);
        run_groups(d, X.dtype, MI_BIAS_ACT_DB, pc, u32(C), op);
    }
}

namespace {

void upfirdn2d_run(const char* op, const Tensor& In, int N, int C, int Hin, int Win, const Tensor& f, int fH, int fW,
                   int up_x, int up_y, int down_x, int down_y, int px0, int px1, int py0, int py1, bool flip,
                   float gain, Tensor& Out) {
    dt_code(In.dtype, op);
    need(op, f.dtype == In.dtype, "f.dtype must match input.dtype");
    need(op, up_x >= 1 && up_y >= 1 && down_x >= 1 && down_y >= 1, "up/down factors must be >= 1");
    need(op, N >= 0 && C >= 0 && Hin >= 0 && Win >= 0, "negative dimension");
    need(op, In.rows == N && In.cols == static_cast<long long>(C) * Hin * Win, "input shape mismatch");
    need(op, f.rows == fH && f.cols == fW && fH > 0 && fW > 0, "filter shape mismatch");
    const int Hp = Hin * up_y + py0 + py1, Wp = Win * up_x + px0 + px1;
    need(op, Hp >= fH && Wp >= fW, "padded input smaller than filter");
    const int Hout = (Hp - fH) / down_y + 1, Wout = (Wp - fW) / down_x + 1;
    ensure(Out, N, static_cast<long long>(C) * Hout * Wout, In.dtype, op);
    MiscPush pc{};
    pc.a = addr(In.data); pc.b = addr(f.data); pc.c = addr(Out.data);
    const std::uint32_t u[] = {u32(static_cast<long long>(N) * C), u32(Hin), u32(Win), u32(Hout), u32(Wout),
                               u32(fH), u32(fW), u32(up_x), u32(up_y), u32(down_x), u32(down_y),
                               static_cast<std::uint32_t>(px0), static_cast<std::uint32_t>(py0), flip ? 1u : 0u, 0, 0};
    std::copy(std::begin(u), std::end(u), pc.u);
    pc.f[0] = gain;
    run_items(device_of(In), In.dtype, MI_UPFIRDN, pc, static_cast<std::uint64_t>(N) * C * Hout * Wout, op);
}

}  // namespace

void upfirdn2d_forward(const Tensor& X, const Tensor& f, int N, int C, int H, int Wd, int fH, int fW, int up_x,
                       int up_y, int down_x, int down_y, int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                       bool flip_filter, float gain, Tensor& Y) {
    upfirdn2d_run("upfirdn2d_forward", X, N, C, H, Wd, f, fH, fW, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0,
                  pad_y1, flip_filter, gain, Y);
}

// The adjoint is the forward with up / down swapped, the flip inverted and
// the padding recomputed (NVlabs _upfirdn2d_cuda, as the CPU does).
void upfirdn2d_backward(const Tensor& dY, const Tensor& f, int N, int C, int H, int Wd, int fH, int fW, int up_x,
                        int up_y, int down_x, int down_y, int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                        bool flip_filter, float gain, Tensor& dX) {
    const char* op = "upfirdn2d_backward";
    need(op, down_x >= 1 && down_y >= 1, "up/down factors must be >= 1");
    const int Hout = (H * up_y + pad_y0 + pad_y1 - fH) / down_y + 1;
    const int Wout = (Wd * up_x + pad_x0 + pad_x1 - fW) / down_x + 1;
    const int p_x0 = fW - pad_x0 - 1, p_x1 = Wd * up_x - Wout * down_x + pad_x0 - up_x + 1;
    const int p_y0 = fH - pad_y0 - 1, p_y1 = H * up_y - Hout * down_y + pad_y0 - up_y + 1;
    upfirdn2d_run(op, dY, N, C, Hout, Wout, f, fH, fW, down_x, down_y, up_x, up_y, p_x0, p_x1, p_y0, p_y1,
                  !flip_filter, gain, dX);
    need(op, dX.rows == N && dX.cols == static_cast<long long>(C) * H * Wd, "internal dX shape mismatch");
}

void fill_vulkan_vtable_misc(::brotensor::detail::OpsVTable& v) {
    v.embedding_lookup_forward = &embedding_lookup_forward;
    v.masked_mean_pool_forward = &masked_mean_pool_forward;
    v.masked_mean_pool_backward = &masked_mean_pool_backward;
    v.build_slot_mask = &build_slot_mask;
    v.build_causal_mask_row = &build_causal_mask_row;
    v.threshold_u8 = &threshold_u8;
    v.rows_count_above = &rows_count_above;
    v.xavier_init = &xavier_init;
    v.sgd_step = &sgd_step;
    v.adam_step = &adam_step;
    v.mse_vec_forward = &mse_vec_forward;
    v.mse_vec_backward = &mse_vec_backward;
    v.mse_vec_per_sample = &mse_vec_per_sample;
    v.attention_token_moments = &attention_token_moments;
    v.bias_act_forward = &bias_act_forward;
    v.bias_act_backward = &bias_act_backward;
    v.upfirdn2d_forward = &upfirdn2d_forward;
    v.upfirdn2d_backward = &upfirdn2d_backward;
}

}  // namespace brotensor::detail::vulkan
