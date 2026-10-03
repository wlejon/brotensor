// Vulkan parity for the flash-attention backwards (ops_fa_bwd.cpp,
// fa_bwd.comp) against the CPU backend: flash_attention_backward (masked,
// causal, cross-length, head widths 16-256), flash_attention_varlen_backward
// (empty and one-row sequences), flash_attention_packed_qkv_backward
// (windowed and not) and flash_attention_qkvo_backward (self and cross, with
// and without biases). 16-bit inputs are rounded to the dtype before both
// backends see them; the CPU computes in FP32. `--only=train`.
//
// Every 16-bit case runs three times: on the automatic kernel (fa_bwd_cm.comp
// on a cooperative-matrix device up to head width 128), on fa_bwd_cm.comp
// without its clustered-layout fast path, and forced onto the FMA kernel
// (fa_bwd.comp); each call checks which kernel ran
// (dv::fa_backward_last_path). BROTENSOR_VK_NO_COOPMAT=1 makes all three FMA.

#include "test_vulkan_common.h"

#include "detail/attention.h"

#include <brotensor/ops/flash_attention.h>
#include <brotensor/vulkan.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vkt {

namespace {

namespace dv = brotensor::detail::vulkan;

int g_mode = 0;   // this round's dv::set_fa_backward_override mode: 0 auto, 1 FMA, 3 coopmat agnostic

// The kernel a backward core should have taken: coopmat for 16-bit up to
// head width 128 on a cooperative-matrix device, unless forced to FMA.
void expect_path(Dtype dt, int hd, const std::string& what) {
    static const bool cm_dev = [] {
        const char* e = std::getenv("BROTENSOR_VK_NO_COOPMAT");
        const bool off = e && *e && std::strcmp(e, "0") != 0;
        return !off && brotensor::vulkan::device_info(vk()).cooperative_matrix;
    }();
    const bool cm = cm_dev && g_mode != 1 && dt != Dtype::FP32 && hd <= 128;
    const std::string got = dv::fa_backward_last_path(), want = cm ? "coopmat" : "fma";
    if (got != want) {
        std::printf("  FAIL  %s: ran the %s kernel, expected %s\n", what.c_str(), got.c_str(), want.c_str());
        ++failures();
    }
}

Tensor cpu(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

Tensor i32(Device d, const std::vector<int32_t>& v, int rows, int cols) {
    Tensor t = Tensor::empty_on(d, rows, cols, Dtype::INT32);
    if (!v.empty()) t.copy_from_host_raw(v.data(), v.size() * 4);
    return t;
}

// Gradient tolerance relative to the largest output: the 16-bit store plus a
// few FP32 roundings.
float rel_for(Dtype dt) { return dt == Dtype::FP32 ? 2e-5f : 3.0f * dtype_eps(dt); }

std::string tag(const char* op, const char* what, Dtype dt, const char* name) {
    return std::string(op) + " " + what + " " + dt_name(dt) + (g_mode == 1 ? " [fma] " : g_mode == 3 ? " [agnostic] " : " ") + name;
}

void test_fa(int Lq, int Lk, int H, int hd, bool causal, bool masked, Dtype dt, const char* name) {
    const int D = H * hd;
    const auto q = random_values(std::size_t(Lq) * D, 1, -1, 1, dt), k = random_values(std::size_t(Lk) * D, 2, -1, 1, dt),
               v = random_values(std::size_t(Lk) * D, 3, -1, 1, dt), g = random_values(std::size_t(Lq) * D, 4, -1, 1, dt);
    std::vector<float> m(Lk, 1.0f);
    for (int i = 0; i < Lk; ++i)
        if ((i * 7 + 3) % 5 == 0) m[i] = 0.0f;
    m[0] = 1.0f;
    Tensor mc = cpu(m, 1, Lk), mv = upload(m, 1, Lk, Dtype::FP32);
    Tensor dQc, dKc, dVc, dQv, dKv, dVv;
    brotensor::flash_attention_backward(cpu(q, Lq, D), cpu(k, Lk, D), cpu(v, Lk, D), cpu(q, Lq, D), cpu(g, Lq, D),
                                        masked ? static_cast<const float*>(mc.data) : nullptr, H, causal, dQc, dKc,
                                        dVc);
    Tensor Qv = upload(q, Lq, D, dt);
    brotensor::flash_attention_backward(Qv, upload(k, Lk, D, dt), upload(v, Lk, D, dt), Qv, upload(g, Lq, D, dt),
                                        masked ? static_cast<const float*>(mv.data) : nullptr, H, causal, dQv, dKv,
                                        dVv);
    expect_path(dt, hd, tag("flash_attention_backward", "kernel", dt, name));
    VKT_CHECK(dQv.dtype == dt && dKv.rows == Lk && dVv.cols == D);
    const float rel = rel_for(dt);
    expect_scaled(download(dQv), dQc.to_host_vector(), rel, tag("flash_attention_backward", "dQ", dt, name));
    expect_scaled(download(dKv), dKc.to_host_vector(), rel, tag("flash_attention_backward", "dK", dt, name));
    expect_scaled(download(dVv), dVc.to_host_vector(), rel, tag("flash_attention_backward", "dV", dt, name));
}

void test_varlen(const std::vector<int>& lens, int H, int hd, bool causal, Dtype dt, const char* name) {
    const int D = H * hd, B = static_cast<int>(lens.size());
    std::vector<int32_t> cu(B + 1, 0);
    for (int b = 0; b < B; ++b) cu[b + 1] = cu[b] + lens[b];
    const int T = cu[B];
    const auto q = random_values(std::size_t(T) * D, 11, -1, 1, dt), k = random_values(std::size_t(T) * D, 12, -1, 1, dt),
               v = random_values(std::size_t(T) * D, 13, -1, 1, dt), g = random_values(std::size_t(T) * D, 14, -1, 1, dt);
    Tensor dQc, dKc, dVc, dQv, dKv, dVv;
    brotensor::flash_attention_varlen_backward(cpu(q, T, D), cpu(k, T, D), cpu(v, T, D), cpu(q, T, D), cpu(g, T, D),
                                               cu.data(), cu.data(), B, 0, 0, H, hd, causal, dQc, dKc, dVc);
    Tensor cuv = i32(vk(), cu, B + 1, 1);
    const int32_t* dcu = static_cast<const int32_t*>(cuv.data);
    Tensor Qv = upload(q, T, D, dt);
    dQv = upload(std::vector<float>(std::size_t(T) * D, 7.0f), T, D, dt);   // must be overwritten
    brotensor::flash_attention_varlen_backward(Qv, upload(k, T, D, dt), upload(v, T, D, dt), Qv, upload(g, T, D, dt),
                                               dcu, dcu, B, 0, 0, H, hd, causal, dQv, dKv, dVv);
    expect_path(dt, hd, tag("flash_attention_varlen_backward", "kernel", dt, name));
    const float rel = rel_for(dt);
    expect_scaled(download(dQv), dQc.to_host_vector(), rel, tag("flash_attention_varlen_backward", "dQ", dt, name));
    expect_scaled(download(dKv), dKc.to_host_vector(), rel, tag("flash_attention_varlen_backward", "dK", dt, name));
    expect_scaled(download(dVv), dVc.to_host_vector(), rel, tag("flash_attention_varlen_backward", "dV", dt, name));
}

void test_packed(const std::vector<int>& lens, int H, int hd, int window, Dtype dt, const char* name) {
    const int D = H * hd;
    std::vector<int32_t> bounds;
    int L = 0;
    for (int n : lens) {
        for (int i = 0; i < n; ++i) {
            bounds.push_back(L);
            bounds.push_back(L + n);
        }
        L += n;
    }
    const auto qkv = random_values(std::size_t(L) * 3 * D, 21, -1, 1, dt), g = random_values(std::size_t(L) * D, 22, -1, 1, dt);
    Tensor bc = i32(Device::cpu(), bounds, L, 2), bv = i32(vk(), bounds, L, 2);
    Tensor dc, dv;
    brotensor::flash_attention_packed_qkv_backward(cpu(qkv, L, 3 * D), cpu(g, L, D), bc, H, window, dc);
    brotensor::flash_attention_packed_qkv_backward(upload(qkv, L, 3 * D, dt), upload(g, L, D, dt), bv, H, window, dv);
    expect_path(dt, hd, tag("flash_attention_packed_qkv_backward", "kernel", dt, name));
    expect_scaled(download(dv), dc.to_host_vector(), rel_for(dt),
                  tag("flash_attention_packed_qkv_backward", "dQKV", dt, name));
}

void test_qkvo(int Lq, int Lk, int D, int Dc, int H, bool cross, bool bias, bool causal, Dtype dt, const char* name) {
    struct P {
        Tensor c, v;
    };
    auto pair = [&](int r, int c, std::uint64_t seed, float lo, float hi) {
        const auto x = random_values(std::size_t(r) * c, seed, lo, hi, dt);
        return P{cpu(x, r, c), upload(x, r, c, dt)};
    };
    const float w = 0.25f;
    P X = pair(Lq, D, 31, -1, 1), C = pair(Lk, Dc, 32, -1, 1), dO = pair(Lq, D, 33, -1, 1);
    P Wq = pair(D, D, 34, -w, w), Wk = pair(D, cross ? Dc : D, 35, -w, w), Wv = pair(D, cross ? Dc : D, 36, -w, w),
      Wo = pair(D, D, 37, -w, w);
    P bq = pair(D, 1, 38, -1, 1), bk = pair(D, 1, 39, -1, 1), bv = pair(D, 1, 40, -1, 1), bo = pair(D, 1, 41, -1, 1);
    // Gradients start non-zero: dW* / db* accumulate.
    P dWq = pair(D, D, 42, -1, 1), dWk = pair(D, cross ? Dc : D, 43, -1, 1), dWv = pair(D, cross ? Dc : D, 44, -1, 1),
      dWo = pair(D, D, 45, -1, 1), dbq = pair(D, 1, 46, -1, 1), dbk = pair(D, 1, 47, -1, 1),
      dbv = pair(D, 1, 48, -1, 1), dbo = pair(D, 1, 49, -1, 1);
    Tensor dXc, dXv, dCc, dCv;
    auto run = [&](bool gpu, Tensor& dX, Tensor& dC) {
        auto s = [&](P& p) -> Tensor& { return gpu ? p.v : p.c; };
        auto b = [&](P& p) -> Tensor* { return bias ? &s(p) : nullptr; };
        brotensor::flash_attention_qkvo_backward(s(X), cross ? &s(C) : nullptr, s(Wq), b(bq), s(Wk), b(bk), s(Wv),
                                                 b(bv), s(Wo), b(bo), nullptr, H, causal, s(dO), dX,
                                                 cross ? &dC : nullptr, s(dWq), b(dbq), s(dWk), b(dbk), s(dWv), b(dbv),
                                                 s(dWo), b(dbo));
    };
    run(false, dXc, dCc);
    run(true, dXv, dCv);
    expect_path(dt, D / H, tag("flash_attention_qkvo_backward", "kernel", dt, name));
    // 16-bit: the projections, the attention output and every gradient are
    // rounded to the dtype between steps (the CPU stays FP32).
    const float rel = dt == Dtype::FP32 ? 1e-4f : 4.0f * dtype_eps(dt);
    const char* op = "flash_attention_qkvo_backward";
    expect_scaled(download(dXv), dXc.to_host_vector(), rel, tag(op, "dX", dt, name));
    if (cross) expect_scaled(download(dCv), dCc.to_host_vector(), rel, tag(op, "dCtx", dt, name));
    expect_scaled(download(dWq.v), dWq.c.to_host_vector(), rel, tag(op, "dWq", dt, name));
    expect_scaled(download(dWk.v), dWk.c.to_host_vector(), rel, tag(op, "dWk", dt, name));
    expect_scaled(download(dWv.v), dWv.c.to_host_vector(), rel, tag(op, "dWv", dt, name));
    expect_scaled(download(dWo.v), dWo.c.to_host_vector(), rel, tag(op, "dWo", dt, name));
    if (bias) {
        expect_scaled(download(dbq.v), dbq.c.to_host_vector(), rel, tag(op, "dbq", dt, name));
        expect_scaled(download(dbo.v), dbo.c.to_host_vector(), rel, tag(op, "dbo", dt, name));
    }
}

}  // namespace

void run_train_fa_tests() {
    std::printf("\n[training: flash-attention backwards]\n");
    for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
        for (int mode : {0, 1, 3}) {
            if (dt == Dtype::FP32 && mode != 0) continue;
            g_mode = mode;
            dv::set_fa_backward_override(mode);
            test_fa(77, 77, 2, 64, false, false, dt, "L 77 H 2 hd 64");
            test_fa(77, 77, 2, 64, true, false, dt, "L 77 H 2 hd 64 causal");
            test_fa(40, 96, 4, 32, false, true, dt, "Lq 40 Lk 96 H 4 hd 32 masked");
            test_fa(130, 130, 2, 128, true, true, dt, "L 130 H 2 hd 128 causal masked");
            test_fa(97, 61, 2, 128, false, false, dt, "Lq 97 Lk 61 H 2 hd 128");
            test_fa(33, 33, 3, 80, false, false, dt, "L 33 H 3 hd 80");
            test_fa(70, 70, 2, 96, true, false, dt, "L 70 H 2 hd 96 causal");
            test_fa(45, 45, 2, 20, true, true, dt, "L 45 H 2 hd 20 causal masked (unaligned rows)");
            test_fa(50, 70, 1, 256, false, false, dt, "Lq 50 Lk 70 H 1 hd 256");
            test_varlen({37, 1, 64, 0, 20}, 2, 64, true, dt, "lens 37/1/64/0/20 H 2 hd 64 causal");
            test_varlen({37, 1, 64, 0, 20}, 2, 128, false, dt, "lens 37/1/64/0/20 H 2 hd 128");
            test_varlen({16, 16, 9}, 4, 16, false, dt, "lens 16/16/9 H 4 hd 16");
            test_varlen({50, 3, 41}, 2, 128, true, dt, "lens 50/3/41 H 2 hd 128 causal");
            test_packed({70, 5, 33}, 4, 64, 0, dt, "lens 70/5/33 H 4 hd 64");
            test_packed({70, 5, 33}, 3, 40, 16, dt, "lens 70/5/33 H 3 hd 40 window 16");
            test_packed({90, 41}, 2, 128, 0, dt, "lens 90/41 H 2 hd 128");
            test_packed({60, 7}, 2, 12, 10, dt, "lens 60/7 H 2 hd 12 window 10");
            test_qkvo(48, 48, 64, 64, 4, false, true, true, dt, "self L 48 D 64 H 4 causal biases");
            test_qkvo(20, 30, 64, 48, 2, true, false, false, dt, "cross Lq 20 Lk 30 D 64 Dc 48 H 2");
            test_qkvo(40, 40, 256, 256, 2, false, false, false, dt, "self L 40 D 256 H 2 (hd 128)");
        }
    }
    g_mode = 0;
    dv::set_fa_backward_override(0);
    test_fa(512, 512, 8, 64, true, false, Dtype::FP16, "L 512 H 8 hd 64 causal");
    test_fa(300, 300, 4, 128, false, true, Dtype::BF16, "L 300 H 4 hd 128 masked");
    test_packed({300, 212}, 8, 64, 0, Dtype::FP16, "lens 300/212 H 8 hd 64 (LayaGrad shape)");
}

}  // namespace vkt
