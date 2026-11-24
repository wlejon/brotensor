#pragma once

// Small shared MIR building blocks used across src/kernels/mir_*.cpp.

#include "../jit/gpu_mir_emit_detail.h"

#include <brass/mir/gpu_opt.hpp>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace brotensor::kernels {

using brass::BasicBlock;
using brass::Function;
using brass::Module;
using brass::Type;
using brass::Value;
using jit::gpu_mir::byte_offset;
using jit::gpu_mir::Ctx;
using jit::gpu_mir::entry_params;
using jit::gpu_mir::load_lanes;
using jit::gpu_mir::log2_exact;
using jit::gpu_mir::store_lanes;

inline int elem_bytes(Dtype dt) {
    switch (dt) {
        case Dtype::FP32:
        case Dtype::INT32: return 4;
        case Dtype::FP16:
        case Dtype::BF16:  return 2;
        default: throw std::runtime_error("brotensor: unsupported MIR kernel dtype");
    }
}

inline Function* finalize_fn(Function* fn) {
    fn->rebuild_cfg_predecessors();
    jit::gpu_mir::optimize_function(*fn);
    return fn;
}

inline Value* ptr_at(Ctx& c, Value* base, Value* elem_idx, int esz) {
    return c.b.build_add(base, byte_offset(c, elem_idx, esz));
}

inline Value* load_elem_dt(Ctx& c, Dtype dt, Value* base, Value* elem_idx) {
    return load_lanes(c, dt, 1, ptr_at(c, base, elem_idx, elem_bytes(dt)))[0];
}

inline void store_elem_dt(Ctx& c, Dtype dt, Value* base, Value* elem_idx, Value* val) {
    store_lanes(c, dt, 1, ptr_at(c, base, elem_idx, elem_bytes(dt)), {val});
}

inline Value* clamp_i32(Ctx& c, Value* v, Value* lo, Value* hi) {
    return c.b.build_select(c.b.build_slt(v, lo), lo, c.b.build_select(c.b.build_sgt(v, hi), hi, v));
}

inline Value* grid_stride_ew(Ctx& c) {
    return c.kb.mul(c.kb.nctaid_x(), c.i32(static_cast<int>(jit::gpu_mir::kEwBlock)));
}

inline Value* vec_ptr_at(Ctx& c, Value* base, Value* vec_idx, int lanes, int esz) {
    Value* idx64 = c.b.build_zext_i64(vec_idx);
    const int bytes = lanes * esz;
    Value* off = (bytes == 1) ? idx64 : c.b.build_shl(idx64, c.kb.const_i64(log2_exact(bytes)));
    return c.b.build_add(base, off);
}

inline Value* col_elem_idx(Ctx& c, Value* vec_idx, int lanes) {
    return (lanes == 1) ? vec_idx : c.b.build_shl(vec_idx, c.i32(log2_exact(lanes)));
}

inline Value* vec_count(Ctx& c, Value* n_elems, int lanes) {
    return (lanes == 1) ? n_elems : c.b.build_lshr(n_elems, c.i32(log2_exact(lanes)));
}

inline void emit_vec_grid_stride(Ctx& c, Value* n_elems, int lanes, std::uint32_t block_x,
                                 const std::function<void(Value* vec_idx)>& body) {
    Value* gid = c.kb.global_tid_x();
    Value* stride = c.kb.mul(c.kb.nctaid_x(), c.i32(static_cast<int>(block_x)));
    c.kb.for_range(gid, vec_count(c, n_elems, lanes), stride, body);
}

template <typename ThenFn, typename ElseFn>
inline std::vector<Value*> emit_if_else_values(
    Ctx& c, Value* cond, const std::vector<Type>& types, ThenFn&& then_fn, ElseFn&& else_fn) {
    BasicBlock* from = c.b.current_block();
    BasicBlock* then_bb = c.b.append_block("if_then");
    BasicBlock* else_bb = c.b.append_block("if_else");
    BasicBlock* join = c.b.append_block("if_join");
    std::vector<Value*> out;
    out.reserve(types.size());
    for (Type t : types) out.push_back(c.b.add_block_param(join, t));

    c.b.position_at_end(from);
    c.b.build_br_if(cond, then_bb, else_bb);

    c.b.position_at_end(then_bb);
    c.b.build_br(join, then_fn());

    c.b.position_at_end(else_bb);
    c.b.build_br(join, else_fn());

    c.b.position_at_end(join);
    return out;
}

}  // namespace brotensor::kernels
