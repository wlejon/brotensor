#include "mir_elementwise.h"
#include "mir_math.h"

#include "../vulkan/shaders/op_codes.h"

#include <brass/mir/gpu_opt.hpp>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::kernels {

namespace {

using namespace brotensor::jit::gpu_mir;
using brass::Function;
using brass::Module;
using brass::Type;
using brass::Value;

int elem_bytes(Dtype dt) {
    switch (dt) {
        case Dtype::FP32: return 4;
        case Dtype::FP16:
        case Dtype::BF16: return 2;
        default: throw std::runtime_error("brotensor: unsupported MIR elementwise dtype");
    }
}

void emit_vec_grid_stride(Ctx& c, Value* n_elems, int lanes,
                          const std::function<void(Value* vec_idx)>& body) {
    Value* gid = c.kb.global_tid_x();
    Value* stride = c.kb.mul(c.kb.nctaid_x(), c.i32(static_cast<int>(kEwBlock)));
    Value* n_vecs = (lanes == 1) ? n_elems : c.b.build_lshr(n_elems, c.i32(log2_exact(lanes)));
    c.kb.for_range(gid, n_vecs, stride, body);
}

Value* ptr_at(Ctx& c, Value* base, Value* elem_idx, int esz) {
    return c.b.build_add(base, byte_offset(c, elem_idx, esz));
}

Value* vec_ptr_at(Ctx& c, Value* base, Value* vec_idx, int lanes, int esz) {
    Value* idx64 = c.b.build_zext_i64(vec_idx);
    const int bytes = lanes * esz;
    Value* off = (bytes == 1) ? idx64 : c.b.build_shl(idx64, c.kb.const_i64(log2_exact(bytes)));
    return c.b.build_add(base, off);
}

Function* finalize_fn(Function* fn) {
    fn->rebuild_cfg_predecessors();
    optimize_function(*fn);
    return fn;
}

Value* unary_lane(Ctx& c, std::uint32_t op, Value* v, Value* a, Value* b_val) {
    switch (op) {
        case UOP_COPY:       return v;
        case UOP_RELU:       return c.kb.fmax(v, c.f32(0.0f));
        case UOP_TANH:       return emit_tanh(c, v);
        case UOP_SIGMOID:    return emit_sigmoid(c, v);
        case UOP_SILU:       return c.b.build_mul(v, emit_sigmoid(c, v));
        case UOP_GELU_TANH:  return emit_gelu_tanh(c, v);
        case UOP_GELU_EXACT: return emit_gelu_exact(c, v);
        case UOP_QUICK_GELU: return c.b.build_mul(v, emit_sigmoid(c, c.b.build_mul(c.f32(1.702f), v)));
        case UOP_EXP:        return emit_exp(c, v);
        case UOP_LOG:        return emit_log(c, v);
        case UOP_SIN:        return emit_sin(c, v);
        case UOP_COS:        return emit_cos(c, v);
        case UOP_RSQRT:      return emit_rsqrt(c, v);
        case UOP_ROUND:      return emit_round_even(c, v);
        case UOP_ELU: {
            Value* neg = c.b.build_mul(a, c.b.build_sub(emit_exp(c, v), c.f32(1.0f)));
            return c.b.build_select(c.b.build_sgt(v, c.f32(0.0f)), v, neg);
        }
        case UOP_LEAKY_RELU:
            return c.b.build_select(c.b.build_sgt(v, c.f32(0.0f)), v, c.b.build_mul(a, v));
        case UOP_ADD_SCALAR: return c.b.build_add(v, a);
        case UOP_SCALE:      return c.b.build_mul(v, a);
        case UOP_CLAMP:      return c.kb.clamp_f32(v, a, b_val);
        default: throw std::runtime_error("brotensor: unknown unary op");
    }
}

Value* binary_lane(Ctx& c, std::uint32_t op, Value* yv, Value* xv, Value* a, Value* b_val) {
    switch (op) {
        case BOP_ADD:   return c.b.build_add(yv, xv);
        case BOP_MUL:   return c.b.build_mul(yv, xv);
        case BOP_DIV:   return c.b.build_sdiv(yv, xv);
        case BOP_AXPBY: return c.b.build_add(c.b.build_mul(a, yv), c.b.build_mul(b_val, xv));
        default: throw std::runtime_error("brotensor: unknown binary op");
    }
}

Value* act_bwd_lane(Ctx& c, std::uint32_t op, Value* v, Value* g, Value* a) {
    switch (op) {
        case GOP_RELU:
            return c.b.build_select(c.b.build_sgt(v, c.f32(0.0f)), g, c.f32(0.0f));
        case GOP_TANH: {
            Value* vc = c.kb.clamp_f32(v, c.f32(-1.0f), c.f32(1.0f));
            return c.b.build_mul(g, c.b.build_sub(c.f32(1.0f), c.b.build_mul(vc, vc)));
        }
        case GOP_SIGMOID: {
            Value* vc = c.kb.clamp_f32(v, c.f32(0.0f), c.f32(1.0f));
            return c.b.build_mul(g, c.b.build_mul(vc, c.b.build_sub(c.f32(1.0f), vc)));
        }
        case GOP_SILU: {
            Value* vc = c.kb.clamp_f32(v, c.f32(-88.0f), c.f32(88.0f));
            Value* s = emit_sigmoid(c, vc);
            Value* inner = c.b.build_fma_f32(vc, c.b.build_sub(c.f32(1.0f), s), c.f32(1.0f));
            return c.b.build_mul(g, c.b.build_mul(s, inner));
        }
        case GOP_GELU_TANH:
            return c.b.build_mul(g, emit_gelu_tanh_grad(c, v));
        case GOP_GELU_EXACT: {
            Value* vc = c.kb.clamp_f32(v, c.f32(-20.0f), c.f32(20.0f));
            Value* cdf = c.b.build_fma_f32(emit_erf(c, c.b.build_mul(vc, c.f32(0.7071067811865475f))),
                                           c.f32(0.5f), c.f32(0.5f));
            Value* pdf = emit_exp(c, c.b.build_mul(c.f32(-0.5f), c.b.build_mul(vc, vc)));
            Value* scale = c.b.build_fma_f32(vc, c.b.build_mul(c.f32(0.3989422804014327f), pdf), cdf);
            return c.b.build_mul(g, scale);
        }
        case GOP_QUICK_GELU: {
            Value* vc = c.kb.clamp_f32(v, c.f32(-50.0f), c.f32(50.0f));
            Value* kx = c.b.build_mul(c.f32(1.702f), vc);
            Value* s = emit_sigmoid(c, kx);
            Value* scale = c.b.build_fma_f32(kx, c.b.build_mul(s, c.b.build_sub(c.f32(1.0f), s)), s);
            return c.b.build_mul(g, scale);
        }
        case GOP_EXP:
            return c.b.build_mul(g, emit_exp(c, c.kb.clamp_f32(v, c.f32(-88.0f), c.f32(88.0f))));
        case GOP_LOG:
            return c.b.build_sdiv(g, c.kb.fmax(v, c.f32(1e-37f)));
        case GOP_SIN:
            return c.b.build_mul(g, emit_cos(c, v));
        case GOP_COS:
            return c.b.build_mul(c.b.build_neg(g), emit_sin(c, v));
        case GOP_RSQRT: {
            Value* v3 = c.b.build_mul(c.b.build_mul(v, v), v);
            return c.b.build_mul(c.f32(-0.5f), c.b.build_mul(g, v3));
        }
        case GOP_ELU: {
            Value* neg = c.b.build_mul(a, emit_exp(c, c.kb.clamp_f32(v, c.f32(-88.0f), c.f32(0.0f))));
            return c.b.build_mul(g, c.b.build_select(c.b.build_sgt(v, c.f32(0.0f)), c.f32(1.0f), neg));
        }
        case GOP_LEAKY_RELU:
            return c.b.build_mul(g, c.b.build_select(c.b.build_sgt(v, c.f32(0.0f)), c.f32(1.0f), a));
        case GOP_PASS:
            return g;
        default:
            throw std::runtime_error("brotensor: unknown act_bwd op");
    }
}

}  // namespace

Function* build_unary_mir(Module& mod, std::uint32_t op_code, Dtype dt, int lanes) {
    Function* fn = mod.create_function(
        "mir_unary", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::f32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    const int esz = elem_bytes(dt);
    emit_vec_grid_stride(c, p[2], lanes, [&](Value* k) {
        Value* x_ptr = vec_ptr_at(c, p[0], k, lanes, esz);
        Value* y_ptr = vec_ptr_at(c, p[1], k, lanes, esz);
        auto xs = load_lanes(c, dt, lanes, x_ptr);
        std::vector<Value*> ys(static_cast<std::size_t>(lanes));
        for (int j = 0; j < lanes; ++j) {
            ys[static_cast<std::size_t>(j)] = unary_lane(c, op_code, xs[static_cast<std::size_t>(j)], p[3], p[4]);
        }
        store_lanes(c, dt, lanes, y_ptr, ys);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_binary_mir(Module& mod, std::uint32_t op_code, Dtype dt_y, Dtype dt_x, int lanes) {
    Function* fn = mod.create_function(
        "mir_binary", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::f32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    const int esz_y = elem_bytes(dt_y);
    const int esz_x = elem_bytes(dt_x);
    emit_vec_grid_stride(c, p[2], lanes, [&](Value* k) {
        Value* y_ptr = vec_ptr_at(c, p[0], k, lanes, esz_y);
        Value* x_ptr = vec_ptr_at(c, p[1], k, lanes, esz_x);
        auto yv = load_lanes(c, dt_y, lanes, y_ptr);
        auto xv = load_lanes(c, dt_x, lanes, x_ptr);
        std::vector<Value*> out(static_cast<std::size_t>(lanes));
        for (int j = 0; j < lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            out[js] = binary_lane(c, op_code, yv[js], xv[js], p[3], p[4]);
        }
        store_lanes(c, dt_y, lanes, y_ptr, out);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_act_bwd_mir(Module& mod, std::uint32_t op_code, Dtype dt, int lanes) {
    Function* fn = mod.create_function(
        "mir_act_bwd", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::f32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    const int esz = elem_bytes(dt);
    emit_vec_grid_stride(c, p[3], lanes, [&](Value* k) {
        Value* dy_ptr = vec_ptr_at(c, p[1], k, lanes, esz);
        Value* dx_ptr = vec_ptr_at(c, p[2], k, lanes, esz);
        auto g = load_lanes(c, dt, lanes, dy_ptr);
        std::vector<Value*> v;
        if (op_code != GOP_PASS) {
            v = load_lanes(c, dt, lanes, vec_ptr_at(c, p[0], k, lanes, esz));
        }
        std::vector<Value*> out(static_cast<std::size_t>(lanes));
        for (int j = 0; j < lanes; ++j) {
            const auto js = static_cast<std::size_t>(j);
            out[js] = act_bwd_lane(c, op_code, op_code == GOP_PASS ? c.f32(0.0f) : v[js], g[js], p[4]);
        }
        store_lanes(c, dt, lanes, dx_ptr, out);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_cast_mir(Module& mod, Dtype src_dt, Dtype dst_dt, int lanes) {
    Function* fn = mod.create_function(
        "mir_cast", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    const int esz_s = elem_bytes(src_dt);
    const int esz_d = elem_bytes(dst_dt);
    emit_vec_grid_stride(c, p[2], lanes, [&](Value* k) {
        auto vals = load_lanes(c, src_dt, lanes, vec_ptr_at(c, p[0], k, lanes, esz_s));
        store_lanes(c, dst_dt, lanes, vec_ptr_at(c, p[1], k, lanes, esz_d), vals);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_bias_mir(Module& mod, std::uint32_t mode, Dtype dt, int lanes) {
    Function* fn = mod.create_function(
        "mir_bias", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    const int esz = elem_bytes(dt);
    const int l = (mode == BIAS_ROW) ? lanes : 1;
    emit_vec_grid_stride(c, p[2], l, [&](Value* k) {
        Value* elem_i = (l == 1) ? k : c.b.build_shl(k, c.i32(log2_exact(l)));
        Value* y_ptr = ptr_at(c, p[0], elem_i, esz);
        Value* j_idx = (mode == BIAS_ROW) ? c.b.build_umod(elem_i, p[3])
                                          : c.b.build_umod(c.b.build_udiv(elem_i, p[4]), p[3]);
        Value* b_ptr = ptr_at(c, p[1], j_idx, esz);
        auto yv = load_lanes(c, dt, l, y_ptr);
        auto bv = load_lanes(c, dt, l, b_ptr);
        for (int q = 0; q < l; ++q) {
            const auto qs = static_cast<std::size_t>(q);
            yv[qs] = c.b.build_add(yv[qs], bv[qs]);
        }
        store_lanes(c, dt, l, y_ptr, yv);
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

Function* build_glu_mir(Module& mod, std::uint32_t op_code, Dtype dt_in, Dtype dt_out, int lanes) {
    Function* fn = mod.create_function(
        "mir_glu", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});
    brass::codegen::KernelBuilder kb(mod, fn);
    Ctx c(kb);
    auto p = entry_params(c, fn);
    const int esz_in = elem_bytes(dt_in);
    const int esz_out = elem_bytes(dt_out);
    const int l = (op_code == GLU_GEGLU_PAIRS) ? 1 : lanes;

    emit_vec_grid_stride(c, p[3], l, [&](Value* k) {
        Value* elem_i = (l == 1) ? k : c.b.build_shl(k, c.i32(log2_exact(l)));
        Value* d = p[4];
        Value* r = c.b.build_udiv(elem_i, d);
        Value* col = c.b.build_umod(elem_i, d);
        Value* row_base = c.b.build_shl(c.b.build_mul(r, d), c.i32(1));  // 2 * r * d

        if (op_code == GLU_GEGLU_PAIRS) {
            Value* idx = c.b.build_add(row_base, c.b.build_shl(col, c.i32(1)));
            auto av = load_lanes(c, dt_in, 1, ptr_at(c, p[0], idx, esz_in));
            auto bv = load_lanes(c, dt_in, 1, ptr_at(c, p[0], c.b.build_add(idx, c.i32(1)), esz_in));
            Value* out = c.b.build_mul(av[0], emit_gelu_exact(c, bv[0]));
            store_lanes(c, dt_out, 1, ptr_at(c, p[2], elem_i, esz_out), {out});
            return;
        }

        Value* ia = c.b.build_add(row_base, col);
        Value* ib = c.b.build_add(ia, d);
        auto a = load_lanes(c, dt_in, l, ptr_at(c, p[0], ia, esz_in));
        auto bv = load_lanes(c, dt_in, l, ptr_at(c, p[0], ib, esz_in));

        if (op_code <= GLU_GEGLU_EXACT) {
            std::vector<Value*> out(static_cast<std::size_t>(l));
            for (int j = 0; j < l; ++j) {
                const auto js = static_cast<std::size_t>(j);
                if (op_code == GLU_SWIGLU) {
                    out[js] = c.b.build_mul(c.b.build_mul(a[js], emit_sigmoid(c, a[js])), bv[js]);
                } else if (op_code == GLU_GEGLU_TANH) {
                    out[js] = c.b.build_mul(a[js], emit_gelu_tanh(c, bv[js]));
                } else {
                    out[js] = c.b.build_mul(a[js], emit_gelu_exact(c, bv[js]));
                }
            }
            store_lanes(c, dt_out, l, ptr_at(c, p[2], elem_i, esz_out), out);
        } else {
            auto g = load_lanes(c, dt_in, l, ptr_at(c, p[1], elem_i, esz_in));
            std::vector<Value*> da(static_cast<std::size_t>(l));
            std::vector<Value*> db(static_cast<std::size_t>(l));
            for (int j = 0; j < l; ++j) {
                const auto js = static_cast<std::size_t>(j);
                Value* aj = a[js];
                Value* bj = bv[js];
                Value* gj = g[js];
                if (op_code == GLU_SWIGLU_BWD) {
                    Value* ac = c.kb.clamp_f32(aj, c.f32(-88.0f), c.f32(88.0f));
                    Value* s = emit_sigmoid(c, ac);
                    Value* silu_grad = c.b.build_mul(s, c.b.build_fma_f32(ac, c.b.build_sub(c.f32(1.0f), s), c.f32(1.0f)));
                    da[js] = c.b.build_mul(gj, c.b.build_mul(bj, silu_grad));
                    db[js] = c.b.build_mul(gj, c.b.build_mul(aj, s));
                } else if (op_code == GLU_GEGLU_TANH_BWD) {
                    da[js] = c.b.build_mul(gj, emit_gelu_tanh(c, bj));
                    db[js] = c.b.build_mul(gj, c.b.build_mul(aj, emit_gelu_tanh_grad(c, bj)));
                } else {
                    Value* bc = c.kb.clamp_f32(bj, c.f32(-20.0f), c.f32(20.0f));
                    Value* cdf = c.b.build_fma_f32(
                        emit_erf(c, c.b.build_mul(bc, c.f32(0.70710678118654752440f))),
                        c.f32(0.5f), c.f32(0.5f));
                    Value* pdf = c.b.build_mul(
                        c.f32(0.39894228040143267794f),
                        emit_exp(c, c.b.build_mul(c.f32(-0.5f), c.b.build_mul(bc, bc))));
                    da[js] = c.b.build_mul(gj, c.b.build_mul(bj, cdf));
                    db[js] = c.b.build_mul(gj, c.b.build_mul(aj, c.b.build_fma_f32(bc, pdf, cdf)));
                }
            }
            store_lanes(c, dt_out, l, ptr_at(c, p[2], ia, esz_out), da);
            store_lanes(c, dt_out, l, ptr_at(c, p[2], ib, esz_out), db);
        }
    });
    c.b.build_ret_void();
    return finalize_fn(fn);
}

}  // namespace brotensor::kernels
