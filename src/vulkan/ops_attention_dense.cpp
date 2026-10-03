// The dense (materialised) attention path, detail/attention.h: per pass of
// heads (and, for very long rows, of query rows),
//   S = Q_h K_h^T                 gemm(), FP32 result whatever the operands
//   P = softmax(scale S [+ bias])  attn_softmax.comp, P in the operands' dtype
//   O_h = P V_h                    gemm(), written into O's head columns
// Q, K, V and O are addressed in place through the GEMM's leading dimensions
// and batch strides (head h of Q is columns [h hd, (h+1) hd) of a row of
// stride ldq, so consecutive heads are a batch stride of hd). GQA runs one
// KV head at a time with K / V broadcast (batch stride 0) over its group.
// Scores never exist below FP32 before the max subtraction: the HIP backend's
// flash_attention_dense.hip makes the same choice for the same reason.

#include "detail/attention.h"
#include "detail/gemm.h"
#include "detail/kernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

struct SoftmaxPush {
    std::uint64_t s, p, bias, mask, qmask;
    std::uint32_t rows, lk, lq, h0, lds, ldp, q0, bhs;
    float scale;
};

std::uint32_t esize(Dtype t) { return t == Dtype::FP32 ? 4u : 2u; }

// FP32 scores per pass: 256 MiB.
constexpr std::size_t kScoreBudget = std::size_t(64) << 20;

}  // namespace

void dense_attention(DeviceCtx& d, const AttnProblem& p, const DenseExtras& x) {
    if (p.lq <= 0) return;
    const int group = p.hq / p.hkv;
    const Dtype dt = p.dt;
    const bool f32 = dt == Dtype::FP32;
    if (x.probs && !f32) throw std::runtime_error(std::string("brotensor: ") + p.op + ": probabilities need FP32");
    const std::uint32_t es = esize(dt);
    const float scale = std::isnan(x.scale) ? static_cast<float>(1.0 / std::sqrt(double(p.hd))) : x.scale;

    const std::size_t per_head = static_cast<std::size_t>(p.lq) * static_cast<std::size_t>(p.lk);
    const int hpp = static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(group == 1 ? p.hq : group,
                                                                                    kScoreBudget / std::max<std::size_t>(per_head, 1))));
    int rows = p.lq;
    if (per_head > kScoreBudget) {
        rows = static_cast<int>(std::max<std::size_t>(1, kScoreBudget / static_cast<std::size_t>(p.lk)));
        if (rows >= 64) rows = rows / 64 * 64;
    }

    const ::brotensor::Device dev = ::brotensor::Device::vulkan(d.index());
    Tensor S, P;
    if (!x.probs) {
        const long long n = static_cast<long long>(rows == p.lq ? hpp : 1) * rows * p.lk;
        S = Tensor::empty_on(dev, static_cast<int>(n), 1, Dtype::FP32);
        if (!f32) P = Tensor::empty_on(dev, static_cast<int>(n), 1, dt);
    }

    const ShaderId sm_id = dt_variant(ShaderId::attn_softmax_f32, dt, p.op);
    const Kernel& ks = d.pipelines().get(sm_id, {x.bias ? 1u : 0u, p.mask ? 1u : 0u, x.qmask ? 1u : 0u,
                                                 x.mask_ge ? 1u : 0u});

    auto pass = [&](int h0, int nh, int r0, int nr) {
        const std::uint64_t kv_off = static_cast<std::uint64_t>(h0 / group) * p.hd * es;
        const long long nrk = static_cast<long long>(nr) * p.lk;
        const std::uint64_t s_addr = x.probs ? x.probs + (static_cast<std::uint64_t>(h0) * p.lq + r0) * p.lk * 4u
                                             : addr(S.data);
        const std::uint64_t p_addr = (x.probs || f32) ? s_addr : addr(P.data);

        GemmArgs g;
        g.op = p.op;
        g.a = p.q + (static_cast<std::uint64_t>(r0) * p.ldq + static_cast<std::uint64_t>(h0) * p.hd) * es;
        g.b = p.k + kv_off;
        g.c = s_addr;
        g.da = g.db = dt;
        g.dc = Dtype::FP32;
        g.m = nr; g.n = p.lk; g.k = p.hd;
        g.lda = p.ldq; g.ldb = p.ldk; g.ldc = p.lk;
        g.batch = nh;
        g.sa = p.hd;
        g.sb = group == 1 ? p.hd : 0;
        g.sc = nrk;
        gemm(d, g);

        const SoftmaxPush sp{s_addr, p_addr, x.bias, p.mask, x.qmask,
                             static_cast<std::uint32_t>(nh * nr), static_cast<std::uint32_t>(p.lk),
                             static_cast<std::uint32_t>(nr), static_cast<std::uint32_t>(h0),
                             static_cast<std::uint32_t>(p.lk), static_cast<std::uint32_t>(p.lk),
                             static_cast<std::uint32_t>(r0),
                             static_cast<std::uint32_t>(x.bias_hs < 0 ? static_cast<long long>(p.lq) * p.lk : x.bias_hs),
                             scale};
        launch(d, ks, sp, std::min<std::uint32_t>(65535, sp.rows));

        GemmArgs o;
        o.op = p.op;
        o.a = p_addr;
        o.b = p.v + kv_off;
        o.c = p.o + (static_cast<std::uint64_t>(r0) * p.ldo + static_cast<std::uint64_t>(h0) * p.hd) * es;
        o.da = (x.probs || f32) ? Dtype::FP32 : dt;
        o.db = o.dc = dt;
        o.m = nr; o.n = p.hd; o.k = p.lk;
        o.lda = p.lk; o.ldb = p.ldk; o.ldc = p.ldo;
        o.nb = true;   // V stored (K = lk, N = hd)
        o.batch = nh;
        o.sa = nrk;
        o.sb = group == 1 ? p.hd : 0;
        o.sc = p.hd;
        gemm(d, o);
    };

    for (int h = 0; h < p.hq;) {
        const int nh = rows < p.lq ? 1 : std::min(hpp, group == 1 ? p.hq - h : group - h % group);
        for (int r0 = 0; r0 < p.lq; r0 += rows) pass(h, nh, r0, std::min(rows, p.lq - r0));
        h += nh;
    }
}

}  // namespace brotensor::detail::vulkan
