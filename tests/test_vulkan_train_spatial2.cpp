// Vulkan parity for the remaining training backwards against the CPU backend:
// the deterministic scatter-adds (scatter_rows_add, embedding_lookup_backward;
// FP32 must match bit for bit), conv_transpose2d_backward_input / _weight,
// and the gather-form adjoints upsample_bilinear_2x_backward,
// interp2d_backward, pad2d_backward, adaptive_avg_pool2d_backward,
// max_pool2d_backward. 16-bit inputs are rounded to the dtype before both
// backends see them. `--only=train`.

#include "test_vulkan_common.h"

#include <brotensor/ops.h>

#include <string>

namespace vkt {

namespace {

Tensor cpu(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

Tensor i32(Device d, const std::vector<int32_t>& v, int rows, int cols) {
    Tensor t = Tensor::empty_on(d, rows, cols, Dtype::INT32);
    if (!v.empty()) t.copy_from_host_raw(v.data(), v.size() * 4);
    return t;
}

std::string tag(const char* op, Dtype dt, const char* name) {
    return std::string(op) + " " + dt_name(dt) + " " + name;
}

// FP32 sums of a few terms: an ulp or two of the largest; 16-bit: the store.
float rel_for(Dtype dt) { return dt == Dtype::FP32 ? 1e-6f : 1.5f * dtype_eps(dt); }

void test_scatter(int M, int R, int C, Dtype dt, const char* name) {
    std::vector<int32_t> idx(M);
    Rng r(5);
    for (int m = 0; m < M; ++m) idx[m] = static_cast<int32_t>(r.next() % (m % 3 == 0 ? 4 : R));   // hot rows
    const auto y = random_values(std::size_t(M) * C, 6, -1, 1, dt);
    Tensor dXc, dXv;
    brotensor::scatter_rows_add(cpu(y, M, C), i32(Device::cpu(), idx, M, 1), R, dXc);
    dXv = upload(std::vector<float>(std::size_t(R) * C, 3.0f), R, C, dt);   // overwritten
    brotensor::scatter_rows_add(upload(y, M, C, dt), i32(vk(), idx, M, 1), R, dXv);
    if (dt == Dtype::FP32) {
        const auto a = download(dXv), b = dXc.to_host_vector();
        expect_equal_bits(a.data(), b.data(), a.size() * 4, tag("scatter_rows_add bit-exact", dt, name));
    } else {
        expect_scaled(download(dXv), dXc.to_host_vector(), rel_for(dt), tag("scatter_rows_add", dt, name));
    }
    // embedding_lookup_backward accumulates into a non-zero table.
    const auto t0 = random_values(std::size_t(R) * C, 7, -1, 1, dt);
    Tensor tc = cpu(t0, R, C), tv = upload(t0, R, C, dt);
    Tensor ix = i32(vk(), idx, M, 1);
    brotensor::embedding_lookup_backward(cpu(y, M, C), idx.data(), M, tc);
    brotensor::embedding_lookup_backward(upload(y, M, C, dt), static_cast<const int32_t*>(ix.data), M, tv);
    if (dt == Dtype::FP32) {
        const auto a = download(tv), b = tc.to_host_vector();
        expect_equal_bits(a.data(), b.data(), a.size() * 4, tag("embedding_lookup_backward bit-exact", dt, name));
    } else {
        expect_scaled(download(tv), tc.to_host_vector(), rel_for(dt), tag("embedding_lookup_backward", dt, name));
    }
}

// An index outside [0, R) is skipped, not a device fault.
void test_scatter_out_of_range() {
    const std::vector<int32_t> idx = {1, -1, 7, 1, 0};
    const std::vector<float> y = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Tensor dX;
    brotensor::scatter_rows_add(upload(y, 5, 2, Dtype::FP32), i32(vk(), idx, 5, 1), 3, dX);
    expect_close(download(dX), {9, 10, 8, 10, 0, 0}, 0, 0, "scatter_rows_add skips out-of-range rows");
}

struct Ct { int N, Ci, H, W, Co, k, s, p, op, d, g; };

void test_conv_t(const Ct& c, Dtype dt, const char* name) {
    const int Ho = (c.H - 1) * c.s - 2 * c.p + c.d * (c.k - 1) + c.op + 1;
    const int Wo = (c.W - 1) * c.s - 2 * c.p + c.d * (c.k - 1) + c.op + 1;
    const int wcols = c.Co / c.g * c.k * c.k;
    const auto w = random_values(std::size_t(c.Ci) * wcols, 8, -0.5f, 0.5f, dt);
    const auto x = random_values(std::size_t(c.N) * c.Ci * c.H * c.W, 9, -1, 1, dt);
    const auto g = random_values(std::size_t(c.N) * c.Co * Ho * Wo, 10, -1, 1, dt);
    const auto w0 = random_values(std::size_t(c.Ci) * wcols, 11, -1, 1, dt);
    Tensor dXc, dXv;
    brotensor::conv_transpose2d_backward_input(cpu(w, c.Ci, wcols), cpu(g, c.N, c.Co * Ho * Wo), c.N, c.Ci, c.H, c.W,
                                               c.Co, c.k, c.k, c.s, c.s, c.p, c.p, c.op, c.op, c.d, c.d, c.g, dXc);
    brotensor::conv_transpose2d_backward_input(upload(w, c.Ci, wcols, dt), upload(g, c.N, c.Co * Ho * Wo, dt), c.N,
                                               c.Ci, c.H, c.W, c.Co, c.k, c.k, c.s, c.s, c.p, c.p, c.op, c.op, c.d,
                                               c.d, c.g, dXv);
    const float rel = dt == Dtype::FP32 ? 1e-5f : 3.0f * dtype_eps(dt);
    expect_scaled(download(dXv), dXc.to_host_vector(), rel, tag("conv_transpose2d_backward_input", dt, name));
    Tensor dWc = cpu(w0, c.Ci, wcols), dWv = upload(w0, c.Ci, wcols, dt);
    brotensor::conv_transpose2d_backward_weight(cpu(x, c.N, c.Ci * c.H * c.W), cpu(g, c.N, c.Co * Ho * Wo), c.N, c.Ci,
                                                c.H, c.W, c.Co, c.k, c.k, c.s, c.s, c.p, c.p, c.op, c.op, c.d, c.d,
                                                c.g, dWc);
    brotensor::conv_transpose2d_backward_weight(upload(x, c.N, c.Ci * c.H * c.W, dt),
                                                upload(g, c.N, c.Co * Ho * Wo, dt), c.N, c.Ci, c.H, c.W, c.Co, c.k,
                                                c.k, c.s, c.s, c.p, c.p, c.op, c.op, c.d, c.d, c.g, dWv);
    expect_scaled(download(dWv), dWc.to_host_vector(), rel, tag("conv_transpose2d_backward_weight", dt, name));
}

void test_resamples(int N, int C, int H, int W, Dtype dt) {
    const std::string nm = std::to_string(N) + "x" + std::to_string(C) + "x" + std::to_string(H) + "x" +
                           std::to_string(W);
    const float rel = rel_for(dt) * 4;
    auto check = [&](const char* op, const Tensor& got, const Tensor& want, const std::string& what) {
        expect_scaled(download(got), want.to_host_vector(), rel, tag(op, dt, (nm + " " + what).c_str()));
    };
    {   // bilinear 2x
        const auto g = random_values(std::size_t(N) * C * 4 * H * W, 20, -1, 1, dt);
        Tensor c, v;
        brotensor::upsample_bilinear_2x_backward(cpu(g, N, C * 4 * H * W), N, C, H, W, c);
        brotensor::upsample_bilinear_2x_backward(upload(g, N, C * 4 * H * W, dt), N, C, H, W, v);
        check("upsample_bilinear_2x_backward", v, c, "");
    }
    struct Io { int ho, wo; };
    for (Io io : {Io{2 * H + 3, W / 2 + 1}, Io{H / 3 + 1, 3 * W}, Io{H, W}}) {
        for (int mode : {0, 1}) {
            const auto g = random_values(std::size_t(N) * C * io.ho * io.wo, 21, -1, 1, dt);
            Tensor c, v;
            brotensor::interp2d_backward(cpu(g, N, C * io.ho * io.wo), N, C, H, W, io.ho, io.wo, mode, c);
            brotensor::interp2d_backward(upload(g, N, C * io.ho * io.wo, dt), N, C, H, W, io.ho, io.wo, mode, v);
            check("interp2d_backward", v, c,
                  std::string(mode ? "bilinear" : "nearest") + " to " + std::to_string(io.ho) + "x" + std::to_string(io.wo));
        }
    }
    VKT_CHECK(throws([&] {
        Tensor v;
        brotensor::interp2d_backward(upload(std::vector<float>(N * C * 4, 1.0f), N, C * 4, dt), N, C, H, W, 2, 2, 2, v);
    }));
    for (int mode : {0, 1, 2}) {
        const int pt = 2, pb = 1, pl = 3, pr = 2;
        const int Hp = H + pt + pb, Wp = W + pl + pr;
        const auto g = random_values(std::size_t(N) * C * Hp * Wp, 22, -1, 1, dt);
        Tensor c, v;
        brotensor::pad2d_backward(cpu(g, N, C * Hp * Wp), N, C, H, W, pt, pb, pl, pr, mode, c);
        brotensor::pad2d_backward(upload(g, N, C * Hp * Wp, dt), N, C, H, W, pt, pb, pl, pr, mode, v);
        check("pad2d_backward", v, c, std::string("mode ") + std::to_string(mode));
    }
    for (Io io : {Io{3, 5}, Io{H, 1}, Io{H + 2, W - 1}}) {
        const auto g = random_values(std::size_t(N) * C * io.ho * io.wo, 23, -1, 1, dt);
        Tensor c, v;
        brotensor::adaptive_avg_pool2d_backward(cpu(g, N, C * io.ho * io.wo), N, C, H, W, io.ho, io.wo, c);
        brotensor::adaptive_avg_pool2d_backward(upload(g, N, C * io.ho * io.wo, dt), N, C, H, W, io.ho, io.wo, v);
        check("adaptive_avg_pool2d_backward", v, c, "to " + std::to_string(io.ho) + "x" + std::to_string(io.wo));
    }
    struct Mp { int k, s, p; };
    for (Mp m : {Mp{2, 2, 0}, Mp{3, 1, 1}, Mp{3, 2, 1}}) {
        const auto x = random_values(std::size_t(N) * C * H * W, 24, -1, 1, Dtype::FP16);   // ties are likely
        Tensor Y, I;
        brotensor::max_pool2d_forward(cpu(x, N, C * H * W), N, C, H, W, m.k, m.k, m.s, m.s, m.p, m.p, Y, I);
        const int ho = (H + 2 * m.p - m.k) / m.s + 1, wo = (W + 2 * m.p - m.k) / m.s + 1;
        std::vector<int32_t> idx(static_cast<std::size_t>(I.size()));
        I.copy_to_host_raw(idx.data(), idx.size() * 4);
        const auto g = random_values(std::size_t(N) * C * ho * wo, 25, -1, 1, dt);
        Tensor c, v;
        brotensor::max_pool2d_backward(cpu(g, N, C * ho * wo), I, N, C, H, W, ho, wo, c);
        brotensor::max_pool2d_backward(upload(g, N, C * ho * wo, dt), i32(vk(), idx, N, C * ho * wo), N, C, H, W, ho,
                                       wo, v);
        if (dt == Dtype::FP32) {
            const auto a = download(v), b = c.to_host_vector();
            expect_equal_bits(a.data(), b.data(), a.size() * 4,
                              tag("max_pool2d_backward bit-exact", dt, (nm + " k" + std::to_string(m.k) + " s" +
                                                                         std::to_string(m.s)).c_str()));
        } else {
            check("max_pool2d_backward", v, c, "k" + std::to_string(m.k) + " s" + std::to_string(m.s));
        }
    }
}

}  // namespace

void run_train_spatial2_tests() {
    std::printf("\n[training: scatter-adds, transposed convolution and resample / pooling backwards]\n");
    for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
        test_scatter(300, 50, 64, dt, "M 300 R 50 C 64 (local sort)");
        test_scatter(5000, 1000, 40, dt, "M 5000 R 1000 C 40 (global sort)");
        test_scatter(37, 151, 768, dt, "M 37 R 151 C 768 (LayaGrad soft rows)");
        test_conv_t({2, 8, 5, 6, 6, 3, 2, 1, 1, 1, 2}, dt, "N 2 8->6 5x6 k3 s2 p1 op1 g2");
        test_conv_t({1, 16, 7, 7, 32, 4, 2, 1, 0, 1, 1}, dt, "N 1 16->32 7x7 k4 s2 p1");
        test_conv_t({2, 4, 6, 5, 4, 3, 1, 1, 1, 2, 1}, dt, "N 2 4->4 6x5 k3 s1 p1 op1 d2 (crop)");
        test_resamples(2, 3, 9, 7, dt);
        test_resamples(1, 2, 32, 40, dt);
    }
    test_scatter_out_of_range();
}

}  // namespace vkt
