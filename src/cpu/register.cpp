// ─── CPU backend registration ──────────────────────────────────────────────
//
// Builds the CPU OpsVTable (only the slots we implement; everything else
// stays nullptr — the dispatcher throws "not implemented on this backend"
// on null lookups), pairs it with the CPU AllocVTable from alloc.cpp, and
// hands them to the registry at static-init time. CPU is therefore always
// available without a prior brotensor::init() call.

#include <brotensor/detail/dispatch.h>
#include <brotensor/tensor.h>

#include <cstdint>

namespace brotensor::detail::cpu {

// ── alloc.cpp ──
const AllocVTable& cpu_alloc_table();

// ── ops_impl.cpp — forward decls of the 16 implemented ops ──
void linear_forward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& b,
                    const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void linear_backward(const ::brotensor::Tensor& W, const ::brotensor::Tensor& x,
                     const ::brotensor::Tensor& dY,
                     ::brotensor::Tensor& dX, ::brotensor::Tensor& dW,
                     ::brotensor::Tensor& dB);
void relu_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void relu_backward(const ::brotensor::Tensor& x, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX);
void tanh_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void tanh_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                   ::brotensor::Tensor& dX);
void sigmoid_forward(const ::brotensor::Tensor& x, ::brotensor::Tensor& y);
void sigmoid_backward(const ::brotensor::Tensor& y, const ::brotensor::Tensor& dY,
                      ::brotensor::Tensor& dX);
void softmax_forward(const ::brotensor::Tensor& logits, ::brotensor::Tensor& probs,
                     const float* mask);
void softmax_backward(const ::brotensor::Tensor& probs,
                      const ::brotensor::Tensor& dProbs,
                      ::brotensor::Tensor& dLogits);
float softmax_xent_segment(const float* lp, const float* tp,
                           float* pp, float* dz,
                           int n, const float* mask);
float softmax_xent(const ::brotensor::Tensor& logits,
                   const ::brotensor::Tensor& target,
                   ::brotensor::Tensor& probs, ::brotensor::Tensor& dLogits,
                   const float* mask);
float mse_scalar(float pred, float target, float& dPred);
void add_inplace(::brotensor::Tensor& y, const ::brotensor::Tensor& x);
void add_scalar_inplace(::brotensor::Tensor& y, float s);
void xavier_init(::brotensor::Tensor& W, uint64_t& rng_state);

} // namespace brotensor::detail::cpu

namespace {

struct CpuStaticRegistrar {
    CpuStaticRegistrar() {
        using namespace ::brotensor;
        using namespace ::brotensor::detail;

        OpsVTable ops{};   // zero-init — every slot nullptr by default

        ops.linear_forward       = &detail::cpu::linear_forward;
        ops.linear_backward      = &detail::cpu::linear_backward;
        ops.relu_forward         = &detail::cpu::relu_forward;
        ops.relu_backward        = &detail::cpu::relu_backward;
        ops.tanh_forward         = &detail::cpu::tanh_forward;
        ops.tanh_backward        = &detail::cpu::tanh_backward;
        ops.sigmoid_forward      = &detail::cpu::sigmoid_forward;
        ops.sigmoid_backward     = &detail::cpu::sigmoid_backward;
        ops.softmax_forward      = &detail::cpu::softmax_forward;
        ops.softmax_backward     = &detail::cpu::softmax_backward;
        ops.softmax_xent         = &detail::cpu::softmax_xent;
        ops.softmax_xent_segment = &detail::cpu::softmax_xent_segment;
        ops.mse_scalar           = &detail::cpu::mse_scalar;
        ops.add_inplace          = &detail::cpu::add_inplace;
        ops.add_scalar_inplace   = &detail::cpu::add_scalar_inplace;
        ops.xavier_init          = &detail::cpu::xavier_init;

        detail::register_backend(Device::CPU, ops,
                                 detail::cpu::cpu_alloc_table());
    }
};

static CpuStaticRegistrar g_cpu_registrar{};

} // anonymous namespace
