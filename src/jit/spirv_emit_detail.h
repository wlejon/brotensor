#pragma once

// Shared MIR-building helpers for the trace JIT's Vulkan emitters
// (spirv_emit_elementwise.cpp, spirv_emit_rownorm.cpp). Everything here is a
// few lines of KernelBuilder MIR; nothing knows SPIR-V syntax.
//
// Values are FP32 scalars, one per lane: a 16-byte load is split into lanes
// with vextract_lane (or word shifts for packed halves), the op chain runs
// per lane, and a store packs the lanes back. The driver's compiler (ACO on
// RADV) keeps the vector loads and stores 128-bit.
//
// BF16 needs the bits of an FP32 value and an FP32 value from bits. MIR has
// no 32-bit bitcast, but brass's SPIR-V target lowers an i32 access of an f32
// shared array as an OpBitcast (spirv_backend_design.md, "Shared memory"), so
// each thread owns one word of a Workgroup array and reinterprets through it:
// store the i32, load the f32 (or the reverse). The slot is the thread's own,
// so no barrier is involved, and the driver forwards the store to the load.

#include "spirv_emit.h"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::jit::spirv::detail {

using brass::Builder;
using brass::Function;
using brass::Type;
using brass::Value;
using brass::codegen::KernelBuilder;

using ValueMap = std::unordered_map<int, std::vector<Value*>>;

struct Ctx {
    KernelBuilder& kb;
    Builder& b;
    // This thread's word of the reinterpretation array, or null when no
    // buffer of the kernel is BF16.
    Value* slot = nullptr;

    explicit Ctx(KernelBuilder& k) : kb(k), b(k.builder()) {}

    Value* i32(int v) { return kb.const_i32(v); }
    Value* f32(float v) { return kb.const_f32(v); }
};

// Creates `fn`'s entry block with one block parameter per kernel parameter.
std::vector<Value*> entry_params(Ctx& c, Function* fn);

// Allocates the reinterpretation array (`threads` words, one per thread of
// the workgroup) and points c.slot at this thread's word. Call in the entry
// block, before any branch.
void alloc_bitcast_slot(Ctx& c, std::uint32_t threads);

bool any_bf16(const plan::ElementwisePlan& p);

// Bit reinterpretation through c.slot.
Value* bits_to_f32(Ctx& c, Value* bits_i32);
Value* f32_to_bits(Ctx& c, Value* x);

// `elem` (i32 element index) as an i64 byte offset for `esz`-byte elements.
Value* byte_offset(Ctx& c, Value* elem, int esz);

// Loads `lanes` consecutive elements of `dt` at `addr` as FP32 values: one
// access (f32, f32x4, u16, i64 = four halves, i32x4 = eight halves).
std::vector<Value*> load_lanes(Ctx& c, Dtype dt, int lanes, Value* addr);
void store_lanes(Ctx& c, Dtype dt, int lanes, Value* addr, const std::vector<Value*>& vals);

// One op on one lane. Throws for an op with no elementwise form.
Value* lane_op(Ctx& c, const TraceNode& n, const std::vector<Value*>& in);
bool has_lane_form(TraceOpKind op);

// Evaluates every node in `order` (default: the whole DAG in creation order,
// which is topological) that has no value yet, `lanes` wide.
ValueMap eval_chain(Ctx& c, const TraceDAG& dag, int lanes, ValueMap vals,
                    const std::vector<int>* order = nullptr);

// brass::target::SpirvTarget::compile with a 1-D workgroup of `block` threads
// (the default the specialisation constants carry).
brass::target::SpirvKernel compile(const Function& fn, std::uint32_t block);

inline int log2_exact(int v) {
    int l = 0;
    while ((1 << l) < v) ++l;
    return l;
}
inline bool is_pow2(int v) { return v > 0 && (v & (v - 1)) == 0; }

}  // namespace brotensor::jit::spirv::detail
