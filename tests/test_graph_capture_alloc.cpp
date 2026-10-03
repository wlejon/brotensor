// Graph capture of a step that allocates: every op inside the captured region
// writes a fresh temporary and frees the last one, the way a model's forward
// pass does. Replay must match an eager run of the same step, for several fresh
// graphs in a row (the first few graphs of a process are where a backend that
// records allocations as graph memory nodes goes wrong). Also covers the lifetimes around a capture: memory that predates the
// capture freed inside it, an output allocated inside it that outlives its
// graph, a block one graph allocated and a second graph's capture frees, and
// that repeated capture/destroy cycles give their memory back.

#include "parity_helpers.h"

#include <brotensor/cuda_graph.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using brotensor::Device;
using brotensor::Tensor;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

struct Net {
    std::vector<Tensor> W, B;
    Tensor Wsq, Bsq;  // one square D x D layer for the lifetime cases
};

// A stack of linear layers of varying width; each layer's output and a second
// "skip" projection are fresh temporaries, the previous layer's output is freed
// as the next one replaces it. Residual-adds into x in place at the end.
static void field_step(Tensor& x, const Net& net) {
    Tensor cur;
    bt::linear_forward_batched(net.W[0], net.B[0], x, cur);
    for (size_t i = 1; i < net.W.size(); ++i) {
        Tensor nxt, skip;
        bt::linear_forward_batched(net.W[i], net.B[i], cur, nxt);
        bt::linear_forward_batched(net.W[i], net.B[i], cur, skip);
        bt::axpby_inplace(nxt, skip, 0.5f, 0.5f);
        cur = std::move(nxt);
    }
    bt::scale_inplace(cur, 0.1f);
    bt::add_inplace(x, cur);
}

int main() {
    bt::init();
    const Device dev = bt_parity::gpu_device();
    if (dev == Device::CPU || dev == Device::Metal) {
        std::printf("no GPU device - skipping\n");
        return 0;
    }
    std::printf("test_graph_capture_alloc on %s\n", bt::to_string(dev).c_str());

    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto rand_vec = [&](size_t n, float s) {
        std::vector<float> v(n);
        for (auto& e : v) e = u(rng) * s;
        return v;
    };

    const int N = 64, D = 256, layers = 200, steps = 8;
    Net net;
    {
        std::vector<int> width(layers + 1, D);
        for (int i = 1; i < layers; ++i) width[i] = 128 + 64 * (i % 7);
        for (int i = 0; i < layers; ++i) {
            const int in = width[i], out = width[i + 1];
            auto w = rand_vec(static_cast<size_t>(out) * in, 1.0f / std::sqrt(float(in)));
            auto b = rand_vec(out, 0.1f);
            net.W.push_back(Tensor::from_host_on(dev, w.data(), out, in));
            net.B.push_back(Tensor::from_host_on(dev, b.data(), out, 1));
        }
        auto w = rand_vec(static_cast<size_t>(D) * D, 1.0f / std::sqrt(float(D)));
        auto b = rand_vec(D, 0.1f);
        net.Wsq = Tensor::from_host_on(dev, w.data(), D, D);
        net.Bsq = Tensor::from_host_on(dev, b.data(), D, 1);
    }
    const std::vector<float> x0 = rand_vec(static_cast<size_t>(N) * D, 1.0f);

    Tensor xr = Tensor::from_host_on(dev, x0.data(), N, D);
    for (int s = 0; s < steps; ++s) field_step(xr, net);
    const std::vector<float> ref = xr.to_host_vector();

    // ── 1. step 0 eager, steps 1..N-1 replayed, fresh graph each round ──────
    for (int round = 0; round < 6; ++round) {
        Tensor x = Tensor::from_host_on(dev, x0.data(), N, D);
        field_step(x, net);
        bt::sync_all();
        bt::CudaGraph g;
        {
            bt::CudaGraphCapture cap;
            field_step(x, net);
            g = cap.finish();
        }
        for (int s = 1; s < steps; ++s) g.launch();
        bt::sync_all();
        const float err = max_abs_diff(x.to_host_vector(), ref);
        std::printf("  round %d  replay vs eager max_err=%g\n", round, err);
        CHECK(err < 1e-4f);
    }

    // ── 2. memory that predates the capture, freed inside it ────────────────
    // The captured kernels read `pre` on every replay, so its buffer must stay
    // put after the free even though later captured temporaries are allocated.
    {
        auto pv = rand_vec(static_cast<size_t>(N) * D, 1.0f);
        Tensor pre = Tensor::from_host_on(dev, pv.data(), N, D);
        Tensor y = Tensor::empty_on(dev, N, D);
        auto step = [&](bool drop_pre) {
            Tensor h;
            bt::linear_forward_batched(net.Wsq, net.Bsq, pre, h);
            if (drop_pre) pre = Tensor();       // freed mid-step
            Tensor h2;
            bt::linear_forward_batched(net.Wsq, net.Bsq, h, h2);  // reuses pre's bytes eagerly
            bt::copy_d2d(h2, 0, y, 0, N * D);
        };
        step(false);
        const std::vector<float> want = y.to_host_vector();
        bt::scale_inplace(y, 0.0f);
        bt::sync_all();
        bt::CudaGraph g;
        {
            bt::CudaGraphCapture cap;
            step(true);
            g = cap.finish();
        }
        for (int r = 0; r < 3; ++r) g.launch();
        bt::sync_all();
        const float err = max_abs_diff(y.to_host_vector(), want);
        std::printf("  pre-capture buffer freed in capture  max_err=%g\n", err);
        CHECK(err < 1e-4f);
    }

    // ── 3. an output allocated inside the capture outlives its graph, and a
    //       second graph whose capture frees it still reads it after the first
    //       graph is gone ──────────────────────────────────────────────────────
    {
        Tensor in = Tensor::from_host_on(dev, x0.data(), N, D);
        Tensor out_a, out_b;
        bt::linear_forward_batched(net.Wsq, net.Bsq, in, out_a);   // warm-up
        const std::vector<float> want_a = out_a.to_host_vector();
        out_a = Tensor();
        bt::CudaGraph ga;
        {
            bt::CudaGraphCapture cap;
            bt::linear_forward_batched(net.Wsq, net.Bsq, in, out_a);  // allocated in capture
            ga = cap.finish();
        }
        ga.launch();
        bt::sync_all();
        CHECK(max_abs_diff(out_a.to_host_vector(), want_a) < 1e-4f);

        Tensor tmp;
        bt::linear_forward_batched(net.Wsq, net.Bsq, out_a, tmp);
        const std::vector<float> want_b = tmp.to_host_vector();
        tmp = Tensor();
        bt::CudaGraph gb;
        {
            bt::CudaGraphCapture cap;
            bt::linear_forward_batched(net.Wsq, net.Bsq, out_a, out_b);
            out_a = Tensor();   // ga's block, freed inside gb's capture
            Tensor scratch = Tensor::empty_on(dev, N, D);
            bt::copy_d2d(out_b, 0, scratch, 0, N * D);
            gb = cap.finish();
        }
        ga.reset();             // gb must keep the block it reads alive
        {
            // churn the allocator: anything that reused the block would clobber it
            std::vector<Tensor> churn;
            for (int i = 0; i < 16; ++i) {
                churn.push_back(Tensor::empty_on(dev, N, D));
                bt::scale_inplace(churn.back(), 0.0f);
            }
        }
        gb.launch();
        bt::sync_all();
        const float err = max_abs_diff(out_b.to_host_vector(), want_b);
        std::printf("  block shared across graphs  max_err=%g\n", err);
        CHECK(err < 1e-4f);
        gb.reset();
        // out_b outlived gb: still readable, then freed through the ordinary path.
        CHECK(max_abs_diff(out_b.to_host_vector(), want_b) < 1e-4f);
    }

    // ── 4. capture/destroy cycles give their memory back ────────────────────
    {
        bt::sync_all();
        std::size_t free0 = 0, total = 0, free1 = 0;
        const bool have_info = bt::device_mem_info(dev, free0, total);
        for (int round = 0; round < 10; ++round) {
            Tensor x = Tensor::from_host_on(dev, x0.data(), N, D);
            bt::CudaGraph g;
            {
                bt::CudaGraphCapture cap;
                field_step(x, net);
                g = cap.finish();
            }
            g.launch();
        }
        bt::sync_all();
        if (have_info && bt::device_mem_info(dev, free1, total)) {
            const double lost_mb = (double(free0) - double(free1)) / (1024.0 * 1024.0);
            std::printf("  10 capture/destroy cycles: %.1f MB not returned\n", lost_mb);
            CHECK(lost_mb < 64.0);
        }
    }

    std::printf("%s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
