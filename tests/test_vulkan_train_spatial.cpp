// Vulkan parity for the spatial training backwards against the CPU backend:
// group_norm_backward and resblock_backward (chunk 9 slots 2), then the
// resample / pad / pooling backwards, scatter-adds and the transposed
// convolution backwards (test_vulkan_train_spatial2.cpp). 16-bit inputs are
// rounded to the dtype before both backends see them; parameter gradients
// start non-zero to check that they accumulate. `--only=train`.

#include "test_vulkan_common.h"

#include <brotensor/ops.h>

#include <string>

namespace vkt {

namespace {

Tensor cpu(const std::vector<float>& v, int rows, int cols) {
    return Tensor::from_host_on(Device::cpu(), v.data(), rows, cols);
}

struct P {
    Tensor c, v;
    std::vector<float> x;
};

P pair(int rows, int cols, std::uint64_t seed, float lo, float hi, Dtype dt) {
    P p;
    p.x = random_values(std::size_t(rows) * cols, seed, lo, hi, dt);
    p.c = cpu(p.x, rows, cols);
    p.v = upload(p.x, rows, cols, dt);
    return p;
}

std::string tag(const char* op, const char* what, Dtype dt, const char* name) {
    return std::string(op) + " " + what + " " + dt_name(dt) + " " + name;
}

void test_group_norm(int N, int C, int H, int W, int G, float offset, Dtype dt, const char* name) {
    const int cols = C * H * W;
    std::vector<float> x = random_values(std::size_t(N) * cols, 70, -1, 1, Dtype::FP32);
    for (float& v : x) v = round_to(dt, v * 2.0f + offset);
    P X{cpu(x, N, cols), upload(x, N, cols, dt), x};
    P g = pair(C, 1, 71, 0.5f, 1.5f, dt), dY = pair(N, cols, 72, -1, 1, dt);
    P dg = pair(C, 1, 73, -1, 1, dt), db = pair(C, 1, 74, -1, 1, dt);
    Tensor dXc, dXv;
    brotensor::group_norm_backward(X.c, g.c, dY.c, N, C, H, W, G, 1e-5f, dXc, dg.c, db.c);
    brotensor::group_norm_backward(X.v, g.v, dY.v, N, C, H, W, G, 1e-5f, dXv, dg.v, db.v);
    const float rel = dt == Dtype::FP32 ? (offset != 0 ? 1e-4f : 2e-5f) : 3.0f * dtype_eps(dt);
    const char* op = "group_norm_backward";
    expect_scaled(download(dXv), dXc.to_host_vector(), rel, tag(op, "dX", dt, name));
    expect_scaled(download(dg.v), dg.c.to_host_vector(), rel, tag(op, "dGamma", dt, name));
    expect_scaled(download(db.v), db.c.to_host_vector(), rel, tag(op, "dBeta", dt, name));
}

void test_resblock(int N, int Ci, int Co, int H, int W, int G, int shift, Dtype dt, const char* name) {
    const int hw = H * W;
    const float w = 0.2f;
    P X = pair(N, Ci * hw, 80, -1, 1, dt), dY = pair(N, Co * hw, 81, -1, 1, dt);
    P g1 = pair(Ci, 1, 82, 0.5f, 1.5f, dt), be1 = pair(Ci, 1, 83, -0.5f, 0.5f, dt);
    P g2 = pair(Co, 1, 84, 0.5f, 1.5f, dt), be2 = pair(Co, 1, 85, -0.5f, 0.5f, dt);
    P W1 = pair(Co, Ci * 9, 86, -w, w, dt), W2 = pair(Co, Co * 9, 87, -w, w, dt), Ws = pair(Co, Ci, 88, -w, w, dt);
    P b1 = pair(Co, 1, 89, -0.5f, 0.5f, dt), b2 = pair(Co, 1, 90, -0.5f, 0.5f, dt), bs = pair(Co, 1, 91, -0.5f, 0.5f, dt);
    P temb = shift == 2 ? pair(N, Co, 92, -1, 1, dt) : pair(Co, 1, 92, -1, 1, dt);
    P dtemb = shift == 2 ? pair(N, Co, 93, -1, 1, dt) : pair(Co, 1, 93, -1, 1, dt);
    P dg1 = pair(Ci, 1, 94, -1, 1, dt), dbe1 = pair(Ci, 1, 95, -1, 1, dt), dg2 = pair(Co, 1, 96, -1, 1, dt),
      dbe2 = pair(Co, 1, 97, -1, 1, dt), dW1 = pair(Co, Ci * 9, 98, -1, 1, dt), dW2 = pair(Co, Co * 9, 99, -1, 1, dt),
      dWs = pair(Co, Ci, 100, -1, 1, dt), db1 = pair(Co, 1, 101, -1, 1, dt), db2 = pair(Co, 1, 102, -1, 1, dt),
      dbs = pair(Co, 1, 103, -1, 1, dt);
    const bool skip = Ci != Co;
    Tensor dXc, dXv;
    auto run = [&](bool gpu, Tensor& dX) {
        auto s = [&](P& p) -> Tensor& { return gpu ? p.v : p.c; };
        brotensor::resblock_backward(s(X), s(g1), s(be1), s(W1), &s(b1), shift ? &s(temb) : nullptr, s(g2), s(be2),
                                     s(W2), &s(b2), skip ? &s(Ws) : nullptr, skip ? &s(bs) : nullptr, N, Ci, Co, H, W,
                                     G, 1e-5f, s(dY), dX, s(dg1), s(dbe1), s(dW1), &s(db1),
                                     shift ? &s(dtemb) : nullptr, s(dg2), s(dbe2), s(dW2), &s(db2),
                                     skip ? &s(dWs) : nullptr, skip ? &s(dbs) : nullptr);
    };
    run(false, dXc);
    run(true, dXv);
    // 16-bit: every intermediate of the composite (GN, SiLU, two convs and
    // their backwards) is rounded to the dtype; the CPU stays FP32.
    const float rel = dt == Dtype::FP32 ? 1e-4f : 4.0f * dtype_eps(dt);
    const char* op = "resblock_backward";
    expect_scaled(download(dXv), dXc.to_host_vector(), rel, tag(op, "dX", dt, name));
    expect_scaled(download(dW1.v), dW1.c.to_host_vector(), rel, tag(op, "dW1", dt, name));
    expect_scaled(download(dW2.v), dW2.c.to_host_vector(), rel, tag(op, "dW2", dt, name));
    expect_scaled(download(dg1.v), dg1.c.to_host_vector(), rel, tag(op, "dGamma1", dt, name));
    expect_scaled(download(dbe1.v), dbe1.c.to_host_vector(), rel, tag(op, "dBeta1", dt, name));
    expect_scaled(download(dg2.v), dg2.c.to_host_vector(), rel, tag(op, "dGamma2", dt, name));
    expect_scaled(download(dbe2.v), dbe2.c.to_host_vector(), rel, tag(op, "dBeta2", dt, name));
    expect_scaled(download(db1.v), db1.c.to_host_vector(), rel, tag(op, "db1", dt, name));
    expect_scaled(download(db2.v), db2.c.to_host_vector(), rel, tag(op, "db2", dt, name));
    if (shift) expect_scaled(download(dtemb.v), dtemb.c.to_host_vector(), rel, tag(op, "dt_emb_shift", dt, name));
    if (skip) {
        expect_scaled(download(dWs.v), dWs.c.to_host_vector(), rel, tag(op, "dWskip", dt, name));
        expect_scaled(download(dbs.v), dbs.c.to_host_vector(), rel, tag(op, "dbskip", dt, name));
    }
}

}  // namespace

void run_train_spatial_tests() {
    std::printf("\n[training: GroupNorm / ResBlock backwards]\n");
    for (Dtype dt : {Dtype::FP32, Dtype::FP16, Dtype::BF16}) {
        test_group_norm(2, 64, 16, 16, 32, 0.0f, dt, "N 2 C 64 16x16 G 32");
        test_group_norm(3, 12, 5, 7, 4, 0.0f, dt, "N 3 C 12 5x7 G 4");
        test_group_norm(1, 512, 1, 120, 1, 0.0f, dt, "N 1 C 512 1x120 G 1 (Kokoro)");
        test_resblock(2, 32, 32, 8, 8, 8, 1, dt, "N 2 C 32 8x8, per-channel shift");
        test_resblock(2, 16, 32, 6, 5, 8, 2, dt, "N 2 C 16->32 6x5, per-sample shift, skip conv");
        test_resblock(1, 64, 64, 16, 16, 32, 0, dt, "N 1 C 64 16x16");
    }
    test_group_norm(2, 32, 9, 9, 8, 50.0f, Dtype::FP32, "N 2 C 32 9x9 G 8, mean 50");
}

}  // namespace vkt
