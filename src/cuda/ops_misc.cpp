// CUDA small misc/utility ops backed by Brass MIR -> PTX (MirKind::Misc &
// MI_UPFIRDN): masked mean pooling, slot/causal masks, threshold_u8,
// rows_count_above, xavier_init, sgd/adam steps, MSE losses,
// attention_token_moments, bias_act, and upfirdn2d.

#include "cu_mir_cache.h"
#include "../kernels/mir_misc.h"
#include "../kernels/mir_vision.h"
#include "../vulkan/shaders/op_codes.h"

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace brotensor::detail::cuda {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

void require_fp(Dtype dt, const char* op) {
    if (dt != Dtype::FP32 && dt != Dtype::FP16 && dt != Dtype::BF16) {
        fail(op, "must be FP32, FP16, or BF16");
    }
}

void need_f32(const char* op, const Tensor& t, const char* name) {
    if (t.dtype != Dtype::FP32) fail(op, std::string(name) + " must be FP32");
}

void need_f32_out(const char* op, const Tensor& t, const char* name) {
    if (t.data != nullptr && t.dtype != Dtype::FP32) {
        fail(op, std::string(name) + " must be FP32");
    }
}

void ensure(Tensor& t, long long r, long long c, Dtype dt, const char* op) {
    if (r < 0 || c < 0 || r * c > 0x7fffffffLL) fail(op, "output too large");
    if (t.rows != r || t.cols != c || t.dtype != dt) {
        t.resize(static_cast<int>(r), static_cast<int>(c), dt);
    }
}

std::uint32_t count32(const Tensor& t, const char* op) {
    const long long n = t.size();
    if (n < 0 || n > 0x7fffffffLL) fail(op, "tensor too large");
    return static_cast<std::uint32_t>(n);
}

std::uintptr_t addr(const void* p) noexcept {
    return reinterpret_cast<std::uintptr_t>(p);
}

std::int32_t i32(long long v) noexcept {
    return static_cast<std::int32_t>(v);
}

void run_misc(std::uint32_t code, Dtype dt, bool has_aux, std::uint64_t items, bool per_group,
              std::span<const std::uintptr_t> ptrs, std::span<const std::int32_t> ints,
              std::span<const float> flts) {
    if (items == 0) return;
    CUfunction fn = mir_function(
        MirKey{MirKind::Misc, code, dt, dt, 1, dt,
               static_cast<std::uint8_t>(has_aux ? 1u : 0u)});
    const std::uint32_t gx = per_group
        ? static_cast<std::uint32_t>(std::min<std::uint64_t>(items, 65535))
        : mir_groups_1d(static_cast<std::uint32_t>(items), 1, kernels::kMiscBlock);
    mir_dispatch_args(fn, gx, kernels::kMiscBlock, ptrs, ints, flts);
}

void check_bias_act(const char* op, const Tensor& X, const Tensor* b, int N, int C, int HW, int act) {
    require_fp(X.dtype, op);
    need(op, act == 0 || act == 1, "act must be 0 (linear) or 1 (lrelu)");
    need(op, N >= 0 && C >= 0 && HW >= 0, "negative dimension");
    need(op, X.rows == N && X.cols == static_cast<long long>(C) * HW, "X shape mismatch");
    if (b && b->data) {
        need(op, b->dtype == X.dtype, "b.dtype must match X.dtype");
        need(op, b->size() == C, "b must have C elements");
    }
}

std::uintptr_t opt_addr(const Tensor* t) noexcept {
    return (t && t->data) ? addr(t->data) : 0;
}

void upfirdn2d_run(const char* op, const Tensor& In, int N, int C, int Hin, int Win,
                   const Tensor& f, int fH, int fW, int up_x, int up_y,
                   int down_x, int down_y, int px0, int px1, int py0, int py1,
                   bool flip, float gain, Tensor& Out) {
    require_fp(In.dtype, op);
    need(op, f.dtype == In.dtype, "f.dtype must match input.dtype");
    need(op, up_x >= 1 && up_y >= 1 && down_x >= 1 && down_y >= 1,
         "up/down factors must be >= 1");
    need(op, N >= 0 && C >= 0 && Hin >= 0 && Win >= 0, "negative dimension");
    need(op, In.rows == N && In.cols == static_cast<long long>(C) * Hin * Win,
         "input shape mismatch");
    need(op, f.rows == fH && f.cols == fW && fH > 0 && fW > 0, "filter shape mismatch");
    const int Hp = Hin * up_y + py0 + py1, Wp = Win * up_x + px0 + px1;
    need(op, Hp >= fH && Wp >= fW, "padded input smaller than filter");
    const int Hout = (Hp - fH) / down_y + 1, Wout = (Wp - fW) / down_x + 1;
    ensure(Out, N, static_cast<long long>(C) * Hout * Wout, In.dtype, op);
    const std::uint64_t items = static_cast<std::uint64_t>(N) * C * Hout * Wout;
    if (items == 0) return;
    CUfunction fn = mir_function(
        MirKey{MirKind::Vision, MI_UPFIRDN, In.dtype, In.dtype, 1, In.dtype,
               static_cast<std::uint8_t>(flip ? 1u : 0u)});
    const std::uintptr_t ptrs[] = {addr(In.data), addr(f.data), addr(Out.data)};
    const std::int32_t ints[] = {
        i32(static_cast<long long>(N) * C), i32(Hin), i32(Win), i32(Hout), i32(Wout),
        i32(fH), i32(fW), i32(up_x), i32(up_y), i32(down_x), i32(down_y),
        i32(px0), i32(py0)};
    const float flts[] = {gain};
    mir_dispatch_args(fn, mir_groups_1d(static_cast<std::uint32_t>(items), 1, kernels::kVisionBlock),
                      kernels::kVisionBlock, ptrs, ints, flts);
}

}  // namespace

void masked_mean_pool_forward(const Tensor& X, const float* d_mask, Tensor& y) {
    constexpr const char* op = "masked_mean_pool_forward";
    require_fp(X.dtype, op);
    ensure(y, X.cols, 1, X.dtype, op);
    if (X.cols == 0) return;
    const std::uintptr_t ptrs[] = {addr(X.data), addr(d_mask), addr(y.data)};
    const std::int32_t ints[] = {i32(X.rows), i32(X.cols)};
    run_misc(MI_MEAN_POOL, X.dtype, d_mask != nullptr,
             static_cast<std::uint64_t>(X.cols), false, ptrs, ints, {});
}

void masked_mean_pool_backward(const Tensor& dY, const float* d_mask, int K, Tensor& dX) {
    constexpr const char* op = "masked_mean_pool_backward";
    require_fp(dY.dtype, op);
    need(op, K >= 0, "K must be >= 0");
    const long long D = dY.size();
    ensure(dX, K, D, dY.dtype, op);
    const std::uintptr_t ptrs[] = {addr(dY.data), addr(d_mask), addr(dX.data)};
    const std::int32_t ints[] = {i32(K), i32(D)};
    run_misc(MI_MEAN_POOL_BWD, dY.dtype, d_mask != nullptr,
             static_cast<std::uint64_t>(K) * D, false, ptrs, ints, {});
}

void build_slot_mask(const Tensor& x, int offset, int K, int stride, Tensor& mask) {
    constexpr const char* op = "build_slot_mask";
    need_f32(op, x, "x");
    need(op, K >= 0 && offset >= 0 && stride >= 0, "offset, K and stride must be >= 0");
    need(op, K == 0 || offset + static_cast<long long>(K - 1) * stride < x.size(),
         "slots reach past x");
    ensure(mask, K, 1, Dtype::FP32, op);
    const std::uintptr_t ptrs[] = {addr(x.data), addr(mask.data)};
    const std::int32_t ints[] = {i32(K), i32(offset), i32(stride)};
    run_misc(MI_SLOT_MASK, Dtype::FP32, false, static_cast<std::uint64_t>(K),
             false, ptrs, ints, {});
}

void build_causal_mask_row(int L, int q, Tensor& mask) {
    constexpr const char* op = "build_causal_mask_row";
    ensure(mask, std::max(L, 0), 1, Dtype::FP32, op);
    if (L <= 0) return;
    const std::uintptr_t ptrs[] = {addr(mask.data)};
    const std::int32_t ints[] = {i32(L), i32(q)};
    run_misc(MI_CAUSAL_ROW, Dtype::FP32, false, static_cast<std::uint64_t>(L),
             false, ptrs, ints, {});
}

void threshold_u8(const Tensor& X, float t, Tensor& Y) {
    constexpr const char* op = "threshold_u8";
    require_fp(X.dtype, op);
    ensure(Y, X.rows, X.cols, Dtype::INT8, op);
    const std::uint32_t n = count32(X, op);
    const std::uintptr_t ptrs[] = {addr(X.data), addr(Y.data)};
    const std::int32_t ints[] = {static_cast<std::int32_t>(n)};
    const float flts[] = {t};
    run_misc(MI_THRESHOLD_U8, X.dtype, false, n, false, ptrs, ints, flts);
}

void rows_count_above(const Tensor& X, float t_lo, float t_hi, Tensor& counts) {
    constexpr const char* op = "rows_count_above";
    require_fp(X.dtype, op);
    ensure(counts, X.rows, 2, Dtype::INT32, op);
    if (X.rows == 0) return;
    if (X.cols == 0) {
        counts.zero();
        return;
    }
    const std::uintptr_t ptrs[] = {addr(X.data), addr(counts.data)};
    const std::int32_t ints[] = {i32(X.rows), i32(X.cols)};
    const float flts[] = {t_lo, t_hi};
    run_misc(MI_COUNT_ABOVE, X.dtype, false, static_cast<std::uint64_t>(X.rows),
             true, ptrs, ints, flts);
}

void xavier_init(Tensor& W, std::uint64_t& rng_state) {
    constexpr const char* op = "xavier_init";
    need_f32(op, W, "W");
    const std::uint32_t n = count32(W, op);
    if (n == 0) return;
    const float limit = std::sqrt(6.0f / static_cast<float>(W.rows + W.cols));
    const std::uintptr_t ptrs[] = {addr(W.data)};
    const std::int32_t ints[] = {
        static_cast<std::int32_t>(n),
        static_cast<std::int32_t>(static_cast<std::uint32_t>(rng_state)),
        static_cast<std::int32_t>(static_cast<std::uint32_t>(rng_state >> 32)),
    };
    const float flts[] = {limit};
    run_misc(MI_XAVIER, Dtype::FP32, false, n, false, ptrs, ints, flts);
    rng_state += static_cast<std::uint64_t>(n) * 0x9E3779B97F4A7C15ULL;
}

void sgd_step(Tensor& param, Tensor& grad, Tensor& velocity, float lr, float momentum) {
    constexpr const char* op = "sgd_step";
    for (const Tensor* t : std::initializer_list<const Tensor*>{&param, &grad, &velocity}) {
        need_f32(op, *t, "param, grad and velocity");
    }
    need(op, grad.size() == param.size() && velocity.size() == param.size(), "size mismatch");
    const std::uint32_t n = count32(param, op);
    const std::uintptr_t ptrs[] = {addr(param.data), addr(grad.data), addr(velocity.data)};
    const std::int32_t ints[] = {static_cast<std::int32_t>(n)};
    const float flts[] = {lr, momentum};
    run_misc(MI_SGD, Dtype::FP32, false, n, false, ptrs, ints, flts);
}

void adam_step(Tensor& param, const Tensor& grad, Tensor& m, Tensor& v,
               float lr, float beta1, float beta2, float eps, int step) {
    constexpr const char* op = "adam_step";
    for (const Tensor* t : std::initializer_list<const Tensor*>{&param, &grad, &m, &v}) {
        need_f32(op, *t, "param, grad, m and v");
    }
    need(op, grad.size() == param.size() && m.size() == param.size() && v.size() == param.size(),
         "size mismatch");
    const float ibc1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
    const float ibc2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));
    const std::uint32_t n = count32(param, op);
    const std::uintptr_t ptrs[] = {addr(param.data), addr(grad.data), addr(m.data), addr(v.data)};
    const std::int32_t ints[] = {static_cast<std::int32_t>(n)};
    const float flts[] = {ibc1, ibc2, lr, beta1, beta2, eps};
    run_misc(MI_ADAM, Dtype::FP32, false, n, false, ptrs, ints, flts);
}

float mse_vec_forward(const Tensor& pred, const Tensor& target) {
    constexpr const char* op = "mse_vec_forward";
    need_f32(op, pred, "pred");
    need_f32(op, target, "target");
    need(op, target.size() == pred.size(), "size mismatch");
    const std::uint32_t n = count32(pred, op);
    if (n == 0) return 0.0f;
    Tensor out = Tensor::empty_on(pred.device, 1, 1, Dtype::FP32);
    CUfunction fn = mir_function(MirKey{MirKind::Misc, MI_MSE_SUM});
    const std::uintptr_t ptrs[] = {addr(pred.data), addr(target.data), addr(out.data)};
    const std::int32_t ints[] = {static_cast<std::int32_t>(n)};
    mir_dispatch_args(fn, 1, kernels::kMiscBlock, ptrs, ints, {});
    float s = 0.0f;
    out.copy_to_host_raw(&s, sizeof(float));
    return s / static_cast<float>(n);
}

void mse_vec_backward(const Tensor& pred, const Tensor& target, Tensor& dPred) {
    constexpr const char* op = "mse_vec_backward";
    need_f32(op, pred, "pred");
    need_f32(op, target, "target");
    need_f32_out(op, dPred, "dPred");
    need(op, target.size() == pred.size(), "size mismatch");
    ensure(dPred, pred.rows, pred.cols, Dtype::FP32, op);
    const std::uint32_t n = count32(pred, op);
    const std::uintptr_t ptrs[] = {addr(pred.data), addr(target.data), addr(dPred.data)};
    const std::int32_t ints[] = {static_cast<std::int32_t>(n)};
    const float flts[] = {n ? 2.0f / static_cast<float>(n) : 0.0f};
    run_misc(MI_MSE_BWD, Dtype::FP32, false, n, false, ptrs, ints, flts);
}

void mse_vec_per_sample(const Tensor& pred, const Tensor& target,
                        Tensor& dPred, Tensor& loss_per_sample) {
    constexpr const char* op = "mse_vec_per_sample";
    need_f32(op, pred, "pred");
    need_f32(op, target, "target");
    need_f32_out(op, dPred, "dPred");
    need_f32_out(op, loss_per_sample, "loss_per_sample");
    need(op, target.size() == pred.size(), "size mismatch");
    ensure(dPred, pred.rows, pred.cols, Dtype::FP32, op);
    ensure(loss_per_sample, pred.size(), 1, Dtype::FP32, op);
    const std::uint32_t n = count32(pred, op);
    const std::uintptr_t ptrs[] = {
        addr(pred.data), addr(target.data), addr(dPred.data), addr(loss_per_sample.data)};
    const std::int32_t ints[] = {static_cast<std::int32_t>(n)};
    run_misc(MI_MSE_SAMPLE, Dtype::FP32, false, n, false, ptrs, ints, {});
}

void attention_token_moments(const Tensor& Attn, int h_lat, int w_lat,
                             Tensor& mass, Tensor& centroid) {
    constexpr const char* op = "attention_token_moments";
    require_fp(Attn.dtype, op);
    need(op, h_lat > 0 && w_lat > 0, "h_lat and w_lat must be positive");
    need(op, Attn.rows == h_lat * w_lat, "Attn.rows must equal h_lat * w_lat");
    const int Lk = Attn.cols;
    ensure(mass, Lk, 1, Dtype::FP32, op);
    ensure(centroid, Lk, 2, Dtype::FP32, op);
    const std::uintptr_t ptrs[] = {addr(Attn.data), addr(mass.data), addr(centroid.data)};
    const std::int32_t ints[] = {i32(Attn.rows), i32(Lk), i32(w_lat)};
    run_misc(MI_MOMENTS, Attn.dtype, false, static_cast<std::uint64_t>(Lk),
             true, ptrs, ints, {});
}

void bias_act_forward(const Tensor& X, const Tensor* b, int N, int C, int HW, int act,
                      float alpha, float gain, float clamp, Tensor& Y) {
    constexpr const char* op = "bias_act_forward";
    check_bias_act(op, X, b, N, C, HW, act);
    ensure(Y, N, static_cast<long long>(C) * HW, X.dtype, op);
    const std::uint32_t n = count32(X, op);
    const std::uintptr_t b_addr = opt_addr(b);
    const std::uintptr_t ptrs[] = {addr(X.data), b_addr, addr(Y.data)};
    const std::int32_t ints[] = {
        static_cast<std::int32_t>(n), i32(C), i32(std::max(HW, 1)), i32(act)};
    const float flts[] = {alpha, gain, clamp};
    run_misc(MI_BIAS_ACT, X.dtype, b_addr != 0, n, false, ptrs, ints, flts);
}

void bias_act_backward(const Tensor& dY, const Tensor& X, const Tensor* b, int N, int C,
                       int HW, int act, float alpha, float gain, float clamp,
                       Tensor& dX, Tensor* dB) {
    constexpr const char* op = "bias_act_backward";
    check_bias_act(op, X, b, N, C, HW, act);
    need(op, dY.dtype == X.dtype, "dY.dtype must match X.dtype");
    need(op, dY.rows == N && dY.cols == X.cols, "dY shape mismatch");
    if (dB) {
        need(op, dB->dtype == X.dtype, "dB.dtype must match X.dtype");
        need(op, dB->size() == C, "dB must have C elements");
    }
    ensure(dX, N, X.cols, X.dtype, op);
    if (N == 0 || X.cols == 0) return;
    const std::uint32_t n = count32(X, op);
    const std::uintptr_t b_addr = opt_addr(b);
    const std::uintptr_t ptrs_bwd[] = {
        addr(dY.data), addr(X.data), b_addr, addr(dX.data)};
    const std::int32_t ints_bwd[] = {
        static_cast<std::int32_t>(n), i32(C), i32(HW), i32(act)};
    const float flts[] = {alpha, gain, clamp};
    run_misc(MI_BIAS_ACT_BWD, X.dtype, b_addr != 0, n, false, ptrs_bwd, ints_bwd, flts);
    if (dB && dB->data) {
        const std::uintptr_t ptrs_db[] = {
            addr(dY.data), addr(X.data), b_addr, addr(dB->data)};
        const std::int32_t ints_db[] = {i32(N), i32(C), i32(HW), i32(act)};
        run_misc(MI_BIAS_ACT_DB, X.dtype, b_addr != 0, static_cast<std::uint64_t>(C),
                 true, ptrs_db, ints_db, flts);
    }
}

void upfirdn2d_forward(const Tensor& X, const Tensor& f, int N, int C, int H, int Wd,
                       int fH, int fW, int up_x, int up_y, int down_x, int down_y,
                       int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                       bool flip_filter, float gain, Tensor& Y) {
    upfirdn2d_run("upfirdn2d_forward", X, N, C, H, Wd, f, fH, fW, up_x, up_y,
                  down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1, flip_filter, gain, Y);
}

void upfirdn2d_backward(const Tensor& dY, const Tensor& f, int N, int C, int H, int Wd,
                        int fH, int fW, int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1,
                        bool flip_filter, float gain, Tensor& dX) {
    constexpr const char* op = "upfirdn2d_backward";
    need(op, down_x >= 1 && down_y >= 1, "up/down factors must be >= 1");
    const int Hout = (H * up_y + pad_y0 + pad_y1 - fH) / down_y + 1;
    const int Wout = (Wd * up_x + pad_x0 + pad_x1 - fW) / down_x + 1;
    const int p_x0 = fW - pad_x0 - 1;
    const int p_x1 = Wd * up_x - Wout * down_x + pad_x0 - up_x + 1;
    const int p_y0 = fH - pad_y0 - 1;
    const int p_y1 = H * up_y - Hout * down_y + pad_y0 - up_y + 1;
    upfirdn2d_run(op, dY, N, C, Hout, Wout, f, fH, fW, down_x, down_y, up_x, up_y,
                  p_x0, p_x1, p_y0, p_y1, !flip_filter, gain, dX);
    need(op, dX.rows == N && dX.cols == static_cast<long long>(C) * H * Wd,
         "internal dX shape mismatch");
}

void fill_cuda_vtable_xavier_init(::brotensor::detail::OpsVTable& v) {
    v.xavier_init = &xavier_init;
}

void fill_cuda_vtable_bias_act(::brotensor::detail::OpsVTable& v) {
    v.bias_act_forward = &bias_act_forward;
    v.bias_act_backward = &bias_act_backward;
}

void fill_cuda_vtable_upfirdn2d(::brotensor::detail::OpsVTable& v) {
    v.upfirdn2d_forward = &upfirdn2d_forward;
    v.upfirdn2d_backward = &upfirdn2d_backward;
}

}  // namespace brotensor::detail::cuda
