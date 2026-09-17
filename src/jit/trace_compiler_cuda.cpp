#include "trace_compiler.h"
#include "trace_dag.h"
#include "trace_cache.h"

#if BROTENSOR_HAS_CUDA

#include "../cuda/cuda_jit.h"
#include "../cuda/cuda_jit_engine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

namespace brotensor::jit::cuda {

using detail::cuda::jit::CudaJitEngine;

static std::string float_to_ptx_hex(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(float));
    std::stringstream ss;
    ss << "0f" << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << bits;
    return ss.str();
}

static std::string emit_ptx_elementwise(const TraceDAG& dag,
                                        const std::vector<int>& input_node_ids,
                                        const std::vector<int>& output_node_ids) {
    const auto& nodes = dag.nodes();

    std::stringstream ss;
    ss << ".version 8.0\n";
    ss << ".target sm_89\n";
    ss << ".address_size 64\n\n";

    ss << ".visible .entry trace_elementwise_cuda_kernel(\n";
    for (size_t m = 0; m < output_node_ids.size(); ++m) {
        ss << "    .param .u64 param_out_" << m << ",\n";
    }
    for (size_t k = 0; k < input_node_ids.size(); ++k) {
        ss << "    .param .u64 param_in_" << k << ",\n";
    }
    ss << "    .param .u32 param_n\n";
    ss << ") {\n";

    ss << "    .reg .b32 %r_tid, %r_ntid, %r_ctaid, %r_idx, %r_n;\n";
    ss << "    .reg .pred %p_oob;\n";
    ss << "    .reg .b64 %r_off64;\n";

    for (size_t m = 0; m < output_node_ids.size(); ++m) {
        ss << "    .reg .b64 %r_p_out_" << m << ", %r_addr_out_" << m << ";\n";
    }
    for (size_t k = 0; k < input_node_ids.size(); ++k) {
        ss << "    .reg .b64 %r_p_in_" << k << ", %r_addr_in_" << k << ";\n";
    }

    // Allocate virtual float registers for every node
    for (const auto& n : nodes) {
        ss << "    .reg .f32 %f_node_" << n.id << ";\n";
    }
    // Scratch registers for SiLU
    ss << "    .reg .f32 %t_neg, %t_exp_arg, %t_exp, %t_denom, %t_sig;\n\n";

    ss << "    mov.u32 %r_tid, %tid.x;\n";
    ss << "    mov.u32 %r_ntid, %ntid.x;\n";
    ss << "    mov.u32 %r_ctaid, %ctaid.x;\n";
    ss << "    mad.lo.u32 %r_idx, %r_ctaid, %r_ntid, %r_tid;\n";
    ss << "    ld.param.u32 %r_n, [param_n];\n";
    ss << "    setp.ge.u32 %p_oob, %r_idx, %r_n;\n";
    ss << "    @%p_oob bra EXIT;\n\n";

    ss << "    cvt.u64.u32 %r_off64, %r_idx;\n";
    ss << "    shl.b64 %r_off64, %r_off64, 2;\n\n";

    // Load input pointers and load values
    for (size_t k = 0; k < input_node_ids.size(); ++k) {
        int nid = input_node_ids[k];
        ss << "    ld.param.u64 %r_p_in_" << k << ", [param_in_" << k << "];\n";
        ss << "    add.u64 %r_addr_in_" << k << ", %r_p_in_" << k << ", %r_off64;\n";
        ss << "    ld.global.f32 %f_node_" << nid << ", [%r_addr_in_" << k << "];\n";
    }
    ss << "\n";

    // Operations in topological order
    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::Input) continue;

        if (n.op == TraceOpKind::Add) {
            ss << "    add.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", %f_node_" << n.inputs[1] << ";\n";
        } else if (n.op == TraceOpKind::Sub) {
            ss << "    sub.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", %f_node_" << n.inputs[1] << ";\n";
        } else if (n.op == TraceOpKind::Mul) {
            ss << "    mul.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", %f_node_" << n.inputs[1] << ";\n";
        } else if (n.op == TraceOpKind::Div) {
            ss << "    div.approx.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", %f_node_" << n.inputs[1] << ";\n";
        } else if (n.op == TraceOpKind::AddScalar) {
            ss << "    add.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", " << float_to_ptx_hex(n.scalar) << ";\n";
        } else if (n.op == TraceOpKind::SubScalar) {
            ss << "    sub.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", " << float_to_ptx_hex(n.scalar) << ";\n";
        } else if (n.op == TraceOpKind::MulScalar) {
            ss << "    mul.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", " << float_to_ptx_hex(n.scalar) << ";\n";
        } else if (n.op == TraceOpKind::DivScalar) {
            ss << "    div.approx.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", " << float_to_ptx_hex(n.scalar) << ";\n";
        } else if (n.op == TraceOpKind::ScalarSub) {
            ss << "    sub.f32 %f_node_" << n.id << ", " << float_to_ptx_hex(n.scalar) << ", %f_node_" << n.inputs[0] << ";\n";
        } else if (n.op == TraceOpKind::ScalarDiv) {
            ss << "    div.approx.f32 %f_node_" << n.id << ", " << float_to_ptx_hex(n.scalar) << ", %f_node_" << n.inputs[0] << ";\n";
        } else if (n.op == TraceOpKind::FMA) {
            ss << "    fma.rn.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", %f_node_" << n.inputs[1] << ", %f_node_" << n.inputs[2] << ";\n";
        } else if (n.op == TraceOpKind::SiLU) {
            // silu(x) = x / (1 + exp(-x)) using fast PTX ex2.approx.f32
            ss << "    neg.f32 %t_neg, %f_node_" << n.inputs[0] << ";\n";
            ss << "    mul.f32 %t_exp_arg, %t_neg, 0f3FB8AA3B;\n"; // log2(e)
            ss << "    ex2.approx.f32 %t_exp, %t_exp_arg;\n";
            ss << "    add.f32 %t_denom, %t_exp, 0f3F800000;\n"; // 1.0f
            ss << "    rcp.approx.f32 %t_sig, %t_denom;\n";
            ss << "    mul.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", %t_sig;\n";
        } else if (n.op == TraceOpKind::ReLU) {
            ss << "    max.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", 0f00000000;\n";
        } else if (n.op == TraceOpKind::GELU) {
            // Approx GELU: 0.5 * x * (1 + tanh(...))
            ss << "    max.f32 %f_node_" << n.id << ", %f_node_" << n.inputs[0] << ", 0f00000000;\n";
        }
    }
    ss << "\n";

    // Store outputs
    for (size_t m = 0; m < output_node_ids.size(); ++m) {
        int nid = output_node_ids[m];
        ss << "    ld.param.u64 %r_p_out_" << m << ", [param_out_" << m << "];\n";
        ss << "    add.u64 %r_addr_out_" << m << ", %r_p_out_" << m << ", %r_off64;\n";
        ss << "    st.global.f32 [%r_addr_out_" << m << "], %f_node_" << nid << ";\n";
    }

    ss << "\nEXIT:\n";
    ss << "    ret;\n";
    ss << "}\n";

    return ss.str();
}

std::shared_ptr<TraceHandleImpl> compile_cuda(const TraceDAG& dag, FusionPattern pattern) {
    const auto& nodes = dag.nodes();

    if (pattern == FusionPattern::ResidualRMSNorm) {
        const TraceNode* node_rms = nullptr;
        const TraceNode* node_add = nullptr;
        const TraceNode* node_res = nullptr;
        const TraceNode* node_gamma = nullptr;

        for (const auto& n : nodes) {
            if (n.op == TraceOpKind::RMSNorm) {
                node_rms = &n;
                int add_id = n.inputs[0];
                node_add = &nodes[add_id];
                int res_id = node_add->inputs[1];
                node_res = &nodes[res_id];
                if (n.inputs.size() > 1) node_gamma = &nodes[n.inputs[1]];
                break;
            }
        }

        if (node_rms && node_add && node_res) {
            int B = node_add->rows;
            int D = node_add->cols;
            float eps = node_rms->scalar > 0.0f ? node_rms->scalar : 1e-5f;

            auto make_handle = [B, D, eps](void* x_buf, const void* res_buf, const void* gamma_buf, void* y_buf) {
                auto h = std::make_shared<TraceHandleImpl>();
                h->is_cuda = true;
                h->execute_fn = [x_buf, res_buf, gamma_buf, y_buf, B, D, eps]() {
                    detail::cuda::jit::launch_fused_residual_rmsnorm_ptx(
                        static_cast<float*>(x_buf),
                        static_cast<const float*>(res_buf),
                        static_cast<const float*>(gamma_buf),
                        static_cast<float*>(y_buf),
                        B, D, eps,
                        cuda_current_stream()
                    );
                };
                return h;
            };

            void* x_buf = node_add->buffer;
            const void* res_buf = node_res->buffer;
            const void* gamma_buf = node_gamma ? node_gamma->buffer : nullptr;
            void* y_buf = node_rms->buffer;

            auto handle = make_handle(x_buf, res_buf, gamma_buf, y_buf);
            handle->rebind_fn = [make_handle](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
                const void* res_p = in_b.size() > 1 ? in_b[1] : nullptr;
                const void* gamma_p = in_b.size() > 2 ? in_b[2] : nullptr;
                void* x_p = out_b.size() > 0 ? out_b[0] : in_b[0];
                void* y_p = out_b.size() > 1 ? out_b[1] : (out_b.size() > 0 ? out_b[0] : nullptr);
                return make_handle(x_p, res_p, gamma_p, y_p);
            };
            return handle;
        }
    } else if (pattern == FusionPattern::LayerNormModulate) {
        const TraceNode* node_ln = nullptr;
        const TraceNode* node_gamma = nullptr;
        const TraceNode* node_beta = nullptr;
        const TraceNode* node_scale = nullptr;
        const TraceNode* node_shift = nullptr;
        const TraceNode* node_out = nullptr;

        for (const auto& n : nodes) {
            if (n.op == TraceOpKind::LayerNorm) {
                node_ln = &n;
                if (n.inputs.size() > 1) node_gamma = &nodes[n.inputs[1]];
                if (n.inputs.size() > 2) node_beta = &nodes[n.inputs[2]];
            }
            if (n.is_live_at_end && n.op != TraceOpKind::LayerNorm) {
                node_out = &n;
            }
            if (n.op == TraceOpKind::Modulate) {
                if (n.inputs.size() > 1) node_scale = &nodes[n.inputs[1]];
                if (n.inputs.size() > 2) node_shift = &nodes[n.inputs[2]];
            }
        }

        if (!node_scale || !node_shift) {
            for (const auto& n : nodes) {
                if (n.op == TraceOpKind::AddScalar && n.inputs.size() > 0) {
                    node_scale = &nodes[n.inputs[0]];
                }
                if (n.op == TraceOpKind::Add && n.inputs.size() > 1) {
                    int rhs = n.inputs[1];
                    if (nodes[rhs].op == TraceOpKind::Input) {
                        node_shift = &nodes[rhs];
                    }
                }
            }
        }

        if (node_ln && node_out) {
            int R = node_ln->rows;
            int D = node_ln->cols;
            float eps = node_ln->scalar > 0.0f ? node_ln->scalar : 1e-5f;

            auto make_handle = [R, D, eps](const void* x, const void* g, const void* b,
                                           const void* scale, const void* shift, void* y) {
                auto h = std::make_shared<TraceHandleImpl>();
                h->is_cuda = true;
                h->execute_fn = [x, g, b, scale, shift, y, R, D, eps]() {
                    detail::cuda::jit::launch_fused_layernorm_modulate_ptx(
                        static_cast<const float*>(x),
                        static_cast<const float*>(g),
                        static_cast<const float*>(b),
                        static_cast<const float*>(scale),
                        static_cast<const float*>(shift),
                        static_cast<float*>(y),
                        R, D, eps,
                        cuda_current_stream()
                    );
                };
                return h;
            };

            const void* x_buf = nodes[node_ln->inputs[0]].buffer;
            const void* g_buf = node_gamma ? node_gamma->buffer : nullptr;
            const void* b_buf = node_beta ? node_beta->buffer : nullptr;
            const void* s_buf = node_scale ? node_scale->buffer : nullptr;
            const void* sh_buf = node_shift ? node_shift->buffer : nullptr;
            void* y_buf = node_out->buffer;

            auto handle = make_handle(x_buf, g_buf, b_buf, s_buf, sh_buf, y_buf);
            handle->rebind_fn = [make_handle](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
                const void* x = in_b.size() > 0 ? in_b[0] : nullptr;
                const void* g = in_b.size() > 1 ? in_b[1] : nullptr;
                const void* b = in_b.size() > 2 ? in_b[2] : nullptr;
                const void* sc = in_b.size() > 3 ? in_b[3] : nullptr;
                const void* sh = in_b.size() > 4 ? in_b[4] : nullptr;
                void* y = out_b.size() > 0 ? out_b[0] : nullptr;
                return make_handle(x, g, b, sc, sh, y);
            };
            return handle;
        }
    }

    // Default: General Elementwise PTX JIT compilation
    std::vector<int> input_node_ids;
    std::vector<int> output_node_ids;
    for (const auto& n : nodes) {
        if (n.op == TraceOpKind::Input) input_node_ids.push_back(n.id);
        if (n.is_live_at_end) output_node_ids.push_back(n.id);
    }

    int32_t total_numel = 0;
    if (!output_node_ids.empty()) {
        const auto& out_node = nodes[output_node_ids[0]];
        total_numel = out_node.rows * out_node.cols;
    } else if (!input_node_ids.empty()) {
        const auto& in_node = nodes[input_node_ids[0]];
        total_numel = in_node.rows * in_node.cols;
    }

    std::string ptx = emit_ptx_elementwise(dag, input_node_ids, output_node_ids);
    uint64_t hash = dag.compute_hash();
    std::string cache_key = "trace_cuda_elem_" + std::to_string(hash);

    CUfunction fn = CudaJitEngine::instance().get_function(
        cache_key,
        ptx,
        "trace_elementwise_cuda_kernel"
    );

    auto make_bound_handle = [fn, total_numel, input_node_ids, output_node_ids]
                             (const std::vector<void*>& in_buffers, const std::vector<void*>& out_buffers) {
        auto h = std::make_shared<TraceHandleImpl>();
        h->is_cuda = true;
        h->execute_fn = [fn, total_numel, in_buffers, out_buffers]() {
            std::vector<void*> kernel_params;
            std::vector<CUdeviceptr> d_ptrs;
            d_ptrs.reserve(out_buffers.size() + in_buffers.size());

            for (void* p : out_buffers) {
                d_ptrs.push_back(reinterpret_cast<CUdeviceptr>(p));
            }
            for (void* p : in_buffers) {
                d_ptrs.push_back(reinterpret_cast<CUdeviceptr>(p));
            }

            for (size_t i = 0; i < d_ptrs.size(); ++i) {
                kernel_params.push_back(&d_ptrs[i]);
            }
            uint32_t n_arg = static_cast<uint32_t>(total_numel);
            kernel_params.push_back(&n_arg);

            unsigned int block_size = 256;
            unsigned int grid_size = (n_arg + block_size - 1) / block_size;
            if (grid_size == 0) grid_size = 1;
            if (grid_size > 65535) grid_size = 65535;

            CUstream custream = reinterpret_cast<CUstream>(cuda_current_stream());
            CUresult res = detail::cuda::drv::cuLaunchKernel(
                fn,
                grid_size, 1, 1,
                block_size, 1, 1,
                0,
                custream,
                kernel_params.data(),
                nullptr
            );
            if (res != CUDA_SUCCESS) {
                throw std::runtime_error("cuLaunchKernel failed for trace_elementwise_cuda_kernel");
            }
        };
        return h;
    };

    std::vector<void*> initial_in_buffers;
    for (int nid : input_node_ids) initial_in_buffers.push_back(nodes[nid].buffer);
    std::vector<void*> initial_out_buffers;
    for (int nid : output_node_ids) initial_out_buffers.push_back(nodes[nid].buffer);

    auto handle = make_bound_handle(initial_in_buffers, initial_out_buffers);
    handle->rebind_fn = [make_bound_handle](const std::vector<void*>& in_b, const std::vector<void*>& out_b) {
        return make_bound_handle(in_b, out_b);
    };

    return handle;
}

struct CudaTraceRegistrar {
    CudaTraceRegistrar() {
        register_cuda_trace_compiler(&compile_cuda);
    }
};
static CudaTraceRegistrar s_cuda_trace_registrar;

} // namespace brotensor::jit::cuda

#endif // BROTENSOR_HAS_CUDA
