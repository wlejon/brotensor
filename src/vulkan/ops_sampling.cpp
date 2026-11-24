// Vulkan token selection: sample_logits / sample_logits_into (temperature,
// top-k, top-p, Philox draw) and the masked-diffusion step
// (masked_diffusion_scores / _commit). Contracts follow the CPU backend
// (src/cpu/sample_logits.cpp, masked_diffusion.cpp): FP32 logits, INT32
// indices, the same Philox / splitmix64 streams, so a draw selects the same
// token as the CPU unless the uniform lands within a few ulp of a
// probability boundary. Kernel: shaders/select.comp, one workgroup per row.
// sample_logits_into reads its base counter on the device and advances it
// there, so it records into a graph without a host round trip.

#include "detail/kernels.h"
#include "vk_mir_cache.h"
#include "../kernels/mir_select.h"

#include <brotensor/detail/dispatch.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brotensor::detail::vulkan {

using ::brotensor::Dtype;
using ::brotensor::Tensor;

namespace {

[[noreturn]] void fail(const char* op, const std::string& why) {
    throw std::runtime_error(std::string("brotensor: ") + op + ": " + why);
}

void need(const char* op, bool ok, const char* what) {
    if (!ok) fail(op, what);
}

std::uint32_t wg_rows(std::uint64_t rows, const char* op) {
    if (rows > 0x7fffffffULL) fail(op, "too many rows");
    return static_cast<std::uint32_t>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(rows, 65535)));
}

void check_sampling(const char* op, const Tensor& logits, float temperature, int top_k, float top_p) {
    need(op, logits.dtype == Dtype::FP32, "logits must be FP32");
    need(op, temperature >= 0.0f, "temperature must be >= 0");
    need(op, top_k >= 0, "top_k must be >= 0");
    need(op, top_p >= 0.0f, "top_p must be >= 0");
    need(op, !(logits.rows > 0 && logits.cols == 0), "vocabulary size (logits.cols) must be > 0");
}

void sample(const char* op, const Tensor& logits, float temperature, int top_k, float top_p, std::uint64_t key,
            std::uint64_t counter, std::uint64_t counter_addr, std::uint64_t scratch, Tensor& indices) {
    const int N = logits.rows, V = logits.cols;
    DeviceCtx& d = device_of(logits);
    Tensor tmp;
    if (scratch == 0 && temperature != 0.0f) {
        tmp = Tensor::empty_on(Device::vulkan(d.index()), N, V, Dtype::FP32);
        scratch = addr(tmp.data);
    }
    const MirKernel mk = mir_kernel(
        d, MirKey{MirKind::SelectSample, 0, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32,
                  static_cast<std::uint8_t>(counter_addr != 0 ? 1u : 0u)});
    const std::uint64_t ptrs[] = {addr(logits.data), scratch, addr(indices.data), counter_addr};
    const std::int32_t ints[] = {
        static_cast<std::int32_t>(static_cast<std::uint32_t>(key)),
        static_cast<std::int32_t>(static_cast<std::uint32_t>(key >> 32)),
        static_cast<std::int32_t>(V),
        static_cast<std::int32_t>(static_cast<std::uint32_t>(counter)),
        static_cast<std::int32_t>(static_cast<std::uint32_t>(counter >> 32)),
        static_cast<std::int32_t>(top_k),
        static_cast<std::int32_t>(N),
    };
    const float flts[] = {temperature, top_p};
    mir_dispatch_args(d, mk, wg_rows(static_cast<std::uint64_t>(N), op), ptrs, ints, flts);
}

void sample_logits(const Tensor& logits, float temperature, int top_k, float top_p, std::uint64_t key,
                   std::uint64_t counter, Tensor& indices) {
    const char* op = "sample_logits";
    check_sampling(op, logits, temperature, top_k, top_p);
    if (indices.rows != logits.rows || indices.cols != 1 || indices.dtype != Dtype::INT32) {
        indices.resize(logits.rows, 1, Dtype::INT32);
    }
    if (logits.rows == 0) return;
    sample(op, logits, temperature, top_k, top_p, key, counter, 0, 0, indices);
}

void sample_logits_into(const Tensor& logits, float temperature, int top_k, float top_p, std::uint64_t key,
                        Tensor& counter, Tensor& scratch, Tensor& indices) {
    const char* op = "sample_logits_into";
    check_sampling(op, logits, temperature, top_k, top_p);
    const int N = logits.rows, V = logits.cols;
    need(op, counter.dtype == Dtype::INT32 && counter.size() >= 1, "counter must be an INT32 tensor with >= 1 element");
    need(op, scratch.dtype == Dtype::FP32 && scratch.size() >= 3LL * N * V,
         "scratch must be FP32 with at least 3*N*V elements");
    need(op, indices.rows == N && indices.cols == 1 && indices.dtype == Dtype::INT32,
         "indices must be a pre-sized (N,1) INT32 tensor");
    if (N == 0) return;
    sample(op, logits, temperature, top_k, top_p, key, 0, addr(counter.data), addr(scratch.data), indices);
    if (temperature != 0.0f) {   // greedy consumes no draws
        DeviceCtx& d = device_of(logits);
        const MirKernel mk = mir_kernel(d, MirKey{MirKind::SelectCounter});
        const std::uint64_t ptrs[] = {addr(counter.data)};
        const std::int32_t ints[] = {N};
        mir_dispatch_args(d, mk, 1, ptrs, ints, {});
    }
}

void masked_diffusion_scores(const Tensor& logits, const Tensor& tokens, int T, int C, int V, int mask_id,
                             float guidance_scale, float layer_penalty, float position_temperature,
                             float class_temperature, float class_top_frac, std::uint64_t seed, Tensor& pred,
                             Tensor& scores, Tensor& confidence) {
    const char* op = "masked_diffusion_scores";
    need(op, logits.dtype == Dtype::FP32, "logits must be FP32");
    need(op, tokens.dtype == Dtype::INT32, "tokens must be INT32");
    need(op, T >= 1 && C >= 1 && V >= 1, "T, C and V must be >= 1");
    need(op, mask_id >= 0 && mask_id < V, "mask_id must be in [0, V)");
    const bool guided = guidance_scale != 0.0f;
    const int R = guided ? 2 * T : T;
    if (logits.rows != R || logits.cols != static_cast<long long>(C) * V) {
        fail(op, "logits must be (" + std::to_string(R) + ", C*V=" + std::to_string(C * V) + ") for T=" +
                     std::to_string(T) + (guided ? " with" : " without") + " guidance");
    }
    need(op, tokens.rows == C && tokens.cols == T, "tokens must be (C, T)");
    if (pred.rows != C || pred.cols != T || pred.dtype != Dtype::INT32) pred.resize(C, T, Dtype::INT32);
    if (scores.rows != C || scores.cols != T || scores.dtype != Dtype::FP32) scores.resize(C, T, Dtype::FP32);
    if (confidence.rows != C || confidence.cols != T || confidence.dtype != Dtype::FP32) {
        confidence.resize(C, T, Dtype::FP32);
    }
    int k_keep = V;
    if (class_temperature > 0.0f) {
        const double kd = std::ceil(static_cast<double>(class_top_frac) * V);
        k_keep = kd < 1.0 ? 1 : (kd > V ? V : static_cast<int>(kd));
    }
    DeviceCtx& d = device_of(logits);
    Tensor lp = Tensor::empty_on(Device::vulkan(d.index()), C * T, V, Dtype::FP32);
    const std::uint8_t flags = static_cast<std::uint8_t>(
        (guided ? 1u : 0u) | (class_temperature > 0.0f ? 2u : 0u) | (position_temperature > 0.0f ? 4u : 0u));
    const MirKernel mk = mir_kernel(
        d, MirKey{MirKind::SelectMd, 0, Dtype::FP32, Dtype::FP32, 1, Dtype::FP32, flags});
    const std::uint64_t ptrs[] = {addr(logits.data), addr(tokens.data), addr(lp.data),
                                  addr(pred.data), addr(scores.data), addr(confidence.data)};
    const std::int32_t ints[] = {
        static_cast<std::int32_t>(static_cast<std::uint32_t>(seed)),
        static_cast<std::int32_t>(static_cast<std::uint32_t>(seed >> 32)),
        T, C, V, mask_id, k_keep,
    };
    const float flts[] = {guidance_scale, layer_penalty, position_temperature, class_temperature};
    mir_dispatch_args(d, mk, wg_rows(static_cast<std::uint64_t>(C) * T, op), ptrs, ints, flts);
}

void masked_diffusion_commit(const Tensor& pred, const Tensor& idx, int k, int step, Tensor& tokens,
                             Tensor& unmask_step) {
    const char* op = "masked_diffusion_commit";
    for (const Tensor* t : {&pred, &idx}) need(op, t->dtype == Dtype::INT32, "pred and idx must be INT32");
    need(op, tokens.dtype == Dtype::INT32 && unmask_step.dtype == Dtype::INT32, "tokens and unmask_step must be INT32");
    need(op, tokens.rows == pred.rows && tokens.cols == pred.cols && unmask_step.rows == pred.rows &&
                 unmask_step.cols == pred.cols, "pred, tokens and unmask_step must share one (C, T) shape");
    need(op, k >= 0, "k must be >= 0");
    need(op, k <= idx.size(), "idx must hold at least k entries");
    if (k == 0) return;
    DeviceCtx& d = device_of(pred);
    const MirKernel mk = mir_kernel(d, MirKey{MirKind::SelectCommit});
    const std::uint64_t ptrs[] = {addr(pred.data), addr(idx.data), addr(tokens.data), addr(unmask_step.data)};
    const std::int32_t ints[] = {k, static_cast<std::int32_t>(pred.size()), step};
    mir_dispatch_args(d, mk, mir_groups_1d(static_cast<std::uint32_t>(k), 1, kernels::kSelectBlock), ptrs, ints, {});
}

}  // namespace

void fill_vulkan_vtable_sampling(::brotensor::detail::OpsVTable& v) {
    v.sample_logits = &sample_logits;
    v.sample_logits_into = &sample_logits_into;
    v.masked_diffusion_scores = &masked_diffusion_scores;
    v.masked_diffusion_commit = &masked_diffusion_commit;
}

}  // namespace brotensor::detail::vulkan
