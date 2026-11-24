// Live Radeon 8060S Vulkan Proof Gate for brass's new GPU primitives:
//   1. FP16 16x16x16 Cooperative-Matrix NT GEMM (C = A @ B^T, double-buffered 128x128x32
//      with bank-padded shared tiles and K-rotation), compared vs linear_forward_batched_ex.
//   2. Hardware sdot4 (SPV_KHR_integer_dot_product) Q8_0 GEMV (y = W_q8_0 * x),
//      compared vs linear_forward_q8_0_fp16 and a double-precision CPU reference.

#include "vulkan_jit.h"

#include <brotensor/kernel_interop.h>
#include <brotensor/ops.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/gpu_opt.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/spirv_target.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace brass;
using brass::codegen::KernelBuilder;
using brass::target::SpirvKernel;
using brass::target::SpirvOptions;
using brass::target::SpirvStage;
using brass::target::SpirvTarget;

namespace dv = brotensor::detail::vulkan;

std::uint16_t f32_to_f16_bits(float x) {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    const std::uint32_t sign = (u >> 16) & 0x8000u;
    const std::int32_t exp = static_cast<std::int32_t>((u >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = u & 0x7FFFFFu;
    if ((u & 0x7F800000u) == 0x7F800000u) {
        return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
        std::uint32_t half_m = mant >> shift;
        const std::uint32_t rem = mant & ((1u << shift) - 1u);
        const std::uint32_t mid = 1u << (shift - 1u);
        if (rem > mid || (rem == mid && (half_m & 1u))) half_m++;
        return static_cast<std::uint16_t>(sign | half_m);
    }
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    std::uint32_t half_e = static_cast<std::uint32_t>(exp) << 10;
    std::uint32_t half_m = mant >> 13;
    const std::uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_m & 1u))) {
        half_m++;
        if (half_m == 0x400u) {
            half_m = 0;
            half_e += 0x400u;
            if (half_e >= 0x7C00u) return static_cast<std::uint16_t>(sign | 0x7C00u);
        }
    }
    return static_cast<std::uint16_t>(sign | half_e | half_m);
}

float f16_bits_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t exp = (static_cast<std::uint32_t>(h) >> 10) & 0x1Fu;
    std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x3FFu;
    std::uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            std::uint32_t e = 127 - 14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3FFu;
            f = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, 4);
    return out;
}

template <typename T>
void set_push(std::vector<std::uint8_t>& buf, const SpirvKernel& sk, std::size_t idx, const T& v) {
    std::memcpy(buf.data() + sk.params[idx].offset, &v, sizeof(T));
}

// ---------------------------------------------------------------------------
// Kernel 1: Double-buffered 128x128x32 FP16 cooperative-matrix NT GEMM
// C(M, N) = A(M, K) @ B(N, K)^T with bank-conflict padded shared tiles and K-rot.
// ---------------------------------------------------------------------------
SpirvKernel build_coopmat_gemm_nt_f16() {
    constexpr std::uint32_t kWg = 128;
    constexpr std::int32_t kSk = 5;    // BK / 8 + 1 = 5 i32x4 (40 fp16) per row
    constexpr std::int32_t kSz = 640;  // 128 * 5 i32x4 per tile buffer

    Module mod("brass_coopmat_gemm_nt_f16");
    Function* fn = mod.create_function(
        "brass_coopmat_gemm_nt_f16", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* A = b.add_block_param(entry, Type::ptr());
    Value* B = b.add_block_param(entry, Type::ptr());
    Value* C = b.add_block_param(entry, Type::ptr());
    Value* M = b.add_block_param(entry, Type::i32());
    Value* N = b.add_block_param(entry, Type::i32());
    Value* K = b.add_block_param(entry, Type::i32());
    (void)M;

    KernelBuilder kb(b);
    Value* As = kb.shared_alloc_i32x4(2 * kSz);
    Value* Bs = kb.shared_alloc_i32x4(2 * kSz);

    Value* tid = kb.tid_x();
    Value* wgx = kb.ctaid_x();
    Value* wgy = kb.ctaid_y();
    Value* sg = kb.subgroup_id();

    Value* m0 = b.build_shl(wgy, kb.const_i32(7));  // wgy * 128
    Value* n0 = b.build_shl(wgx, kb.const_i32(7));  // wgx * 128
    Value* wm0 = b.build_shl(b.build_and(sg, kb.const_i32(1)), kb.const_i32(6));
    Value* wn0 = b.build_shl(b.build_lshr(sg, kb.const_i32(1)), kb.const_i32(6));

    Value* nk = b.build_lshr(K, kb.const_i32(5));  // K / 32
    Value* krot_raw = kb.mul(kb.add(wgx, kb.mul(wgy, kb.const_i32(5))), kb.const_i32(37));
    Value* krot = b.build_umod(krot_raw, nk);

    Value* fa[4];
    Value* fb[4];
    Value* acc[16];
    for (int i = 0; i < 4; ++i) {
        fa[i] = kb.coop_alloc_a();
        fb[i] = kb.coop_alloc_b();
    }
    Value* zero_f = kb.const_f32(0.0f);
    for (int i = 0; i < 16; ++i) {
        acc[i] = kb.coop_alloc_acc();
        kb.coop_fill_acc(acc[i], zero_f);
    }

    Value* base_ptr_A[4];
    Value* base_ptr_B[4];
    Value* smem_off[4];
    Value* c_two_64 = kb.const_i64(2);
    for (int j = 0; j < 4; ++j) {
        Value* idx = kb.add(tid, kb.const_i32(j * 128));
        Value* r = b.build_lshr(idx, kb.const_i32(2));
        Value* q = b.build_and(idx, kb.const_i32(3));
        Value* col_k = b.build_shl(q, kb.const_i32(3));
        Value* a_elems = kb.add(kb.mul(kb.add(m0, r), K), col_k);
        Value* b_elems = kb.add(kb.mul(kb.add(n0, r), K), col_k);
        base_ptr_A[j] = kb.add(A, b.build_mul(b.build_sext_i64(a_elems), c_two_64));
        base_ptr_B[j] = kb.add(B, b.build_mul(b.build_sext_i64(b_elems), c_two_64));
        smem_off[j] = kb.add(kb.mul(r, kb.const_i32(kSk)), q);
    }

    struct LoadedTiles {
        Value* ra[4];
        Value* rb[4];
    };
    auto load_tiles = [&](Value* k0) {
        LoadedTiles t{};
        Value* k0_bytes = b.build_shl(b.build_sext_i64(k0), kb.const_i64(1));
        for (int j = 0; j < 4; ++j) {
            t.ra[j] = b.build_vload(Type::i32x4(), kb.add(base_ptr_A[j], k0_bytes));
            t.rb[j] = b.build_vload(Type::i32x4(), kb.add(base_ptr_B[j], k0_bytes));
        }
        return t;
    };
    auto store_tiles = [&](Value* buf, const LoadedTiles& t) {
        Value* base = kb.mul(buf, kb.const_i32(kSz));
        for (int j = 0; j < 4; ++j) {
            Value* off = kb.add(base, smem_off[j]);
            kb.shared_store_i32x4_indexed(As, off, t.ra[j]);
            kb.shared_store_i32x4_indexed(Bs, off, t.rb[j]);
        }
    };
    Value* c_stride_40 = kb.const_i32(kSk * 8);
    Value* c4_shift = kb.const_i32(4);  // i32x4 element index -> byte offset for add(shared_ptr, off)
    auto mma_tile = [&](Value* buf) {
        Value* base = kb.mul(buf, kb.const_i32(kSz));
        for (int kk = 0; kk < 2; ++kk) {
            Value* kk_off = kb.const_i32(kk * 2);
            for (int i = 0; i < 4; ++i) {
                Value* row_off = kb.mul(kb.add(wm0, kb.const_i32(i * 16)), kb.const_i32(kSk));
                Value* a_off = b.build_shl(kb.add(base, kb.add(row_off, kk_off)), c4_shift);
                kb.coop_load_a(fa[i], kb.add(As, a_off), c_stride_40, false);
            }
            for (int j = 0; j < 4; ++j) {
                Value* row_off = kb.mul(kb.add(wn0, kb.const_i32(j * 16)), kb.const_i32(kSk));
                Value* b_off = b.build_shl(kb.add(base, kb.add(row_off, kk_off)), c4_shift);
                kb.coop_load_b(fb[j], kb.add(Bs, b_off), c_stride_40, true);
            }
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    kb.coop_mma(acc[i * 4 + j], fa[i], fb[j]);
                }
            }
        }
    };

    // Initial prefetch into shared buffer 0.
    store_tiles(kb.const_i32(0), load_tiles(b.build_shl(krot, kb.const_i32(5))));
    kb.sync();

    Value* nk_m1 = kb.sub(nk, kb.const_i32(1));
    kb.for_range(kb.const_i32(0), nk_m1, kb.const_i32(1), [&](Value* kt) {
        Value* cur = b.build_and(kt, kb.const_i32(1));
        Value* nxt = b.build_xor(cur, kb.const_i32(1));
        Value* k_nxt = b.build_shl(b.build_umod(kb.add(kb.add(kt, kb.const_i32(1)), krot), nk),
                                   kb.const_i32(5));
        LoadedTiles next_t = load_tiles(k_nxt);
        mma_tile(cur);
        store_tiles(nxt, next_t);
        kb.sync();
    });

    // Final tile.
    mma_tile(b.build_and(nk_m1, kb.const_i32(1)));

    // Epilogue: store 16x16 FP16 tiles directly to C(M, N) with stride N.
    for (int i = 0; i < 4; ++i) {
        Value* gm = kb.add(kb.add(m0, wm0), kb.const_i32(i * 16));
        Value* row_bytes = b.build_shl(b.build_sext_i64(kb.mul(gm, N)), kb.const_i64(1));
        Value* C_row = kb.add(C, row_bytes);
        for (int j = 0; j < 4; ++j) {
            Value* gn = kb.add(kb.add(n0, wn0), kb.const_i32(j * 16));
            Value* C_ptr = kb.add(C_row, b.build_shl(b.build_sext_i64(gn), kb.const_i64(1)));
            kb.coop_store_acc(acc[i * 4 + j], C_ptr, N, false, true);
        }
    }

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    brass::GpuOptOptions gpu_opts;
    brass::optimize_gpu_function(*fn, mod, gpu_opts);

    SpirvOptions spv_opts;
    spv_opts.stage = SpirvStage::Compute;
    spv_opts.local_size_x = kWg;
    spv_opts.local_size_spec_constants = true;
    return SpirvTarget::compile(*fn, spv_opts);
}

// ---------------------------------------------------------------------------
// Kernel 2a: Quantize FP16 x[K] -> Q8_1-like blocks of 36 bytes:
//   [f32 d, 8 x i32 packed int8[32]]  (1 subgroup of 32 lanes per block)
// ---------------------------------------------------------------------------
SpirvKernel build_quant_x_q8() {
    Module mod("brass_quant_x_q8");
    Function* fn = mod.create_function("brass_quant_x_q8", Type::void_type(), {Type::ptr(), Type::ptr()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x_f16 = b.add_block_param(entry, Type::ptr());
    Value* x_q8 = b.add_block_param(entry, Type::ptr());

    KernelBuilder kb(b);
    Value* blk = kb.ctaid_x();
    Value* lane = kb.lane_id();  // 0..31 under req_subgroup = 32
    Value* elem_idx = kb.add(b.build_shl(blk, kb.const_i32(5)), lane);
    Value* x_ptr = kb.add(x_f16, b.build_shl(b.build_sext_i64(elem_idx), kb.const_i64(1)));
    Value* xf = kb.f16_to_f32(kb.load_u16(x_ptr));

    Value* amax = kb.fabs(xf);
    for (std::uint32_t d : {16u, 8u, 4u, 2u, 1u}) {
        amax = kb.fmax(amax, kb.shfl_bfly_f32(amax, d));
    }
    Value* dx = kb.div_approx(amax, kb.const_f32(127.0f));
    Value* pos = b.build_sgt(amax, kb.const_f32(0.0f));
    Value* inv_dx = b.build_select(pos, kb.div_approx(kb.const_f32(127.0f), amax), kb.const_f32(0.0f));

    Value* scaled = kb.clamp_f32(kb.mul(xf, inv_dx), kb.const_f32(-127.0f), kb.const_f32(127.0f));
    Value* half_s = b.build_select(b.build_sge(scaled, kb.const_f32(0.0f)),
                                   kb.const_f32(0.5f), kb.const_f32(-0.5f));
    Value* qi = kb.f32_to_i32(kb.add(scaled, half_s));
    Value* b0 = b.build_and(qi, kb.const_i32(0xFF));
    Value* b1 = kb.shfl_bfly_i32(b0, 1);
    Value* b2 = kb.shfl_bfly_i32(b0, 2);
    Value* b3 = kb.shfl_bfly_i32(b0, 3);
    Value* pk = b.build_or(b.build_or(b0, b.build_shl(b1, kb.const_i32(8))),
                           b.build_or(b.build_shl(b2, kb.const_i32(16)), b.build_shl(b3, kb.const_i32(24))));

    Value* blk_ptr = kb.add(x_q8, b.build_sext_i64(kb.mul(blk, kb.const_i32(36))));
    kb.if_then(b.build_eq(lane, kb.const_i32(0)), [&]() { kb.store_f32(blk_ptr, dx); });
    kb.if_then(b.build_eq(b.build_and(lane, kb.const_i32(3)), kb.const_i32(0)), [&]() {
        Value* word_off = kb.add(kb.const_i32(4), lane);  // 4 + (lane >> 2) * 4 == 4 + lane
        kb.store_i32(kb.add(blk_ptr, b.build_sext_i64(word_off)), pk);
    });

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    SpirvOptions spv_opts;
    spv_opts.stage = SpirvStage::Compute;
    spv_opts.local_size_x = 32;
    spv_opts.local_size_spec_constants = true;
    return SpirvTarget::compile(*fn, spv_opts);
}

// ---------------------------------------------------------------------------
// Kernel 2b: Hardware sdot4 Q8_0 GEMV on Wave64 (y = W_q8_0 * x)
// Reads pairs of Q8_0 blocks (64 weights = 68 bytes = 17 aligned i32s per lane),
// runs 16x ptx_sdot4 against x_q8, and reduces across 64 lanes.
// ---------------------------------------------------------------------------
SpirvKernel build_gemv_q8_sdot4() {
    constexpr std::uint32_t kWg = 256;

    Module mod("brass_gemv_q8_sdot4");
    Function* fn = mod.create_function(
        "brass_gemv_q8_sdot4", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x_q8 = b.add_block_param(entry, Type::ptr());
    Value* W = b.add_block_param(entry, Type::ptr());
    Value* y = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i32());
    Value* K = b.add_block_param(entry, Type::i32());
    Value* rowbytes = b.add_block_param(entry, Type::i32());

    KernelBuilder kb(b);
    Value* tid = kb.tid_x();
    Value* sg = b.build_lshr(tid, kb.const_i32(6));          // tid / 64 under Wave64
    Value* lane = b.build_and(tid, kb.const_i32(63));        // tid % 64
    Value* row = kb.add(b.build_shl(kb.ctaid_x(), kb.const_i32(2)), sg);
    Value* in_bounds = b.build_slt(row, N);

    BasicBlock* work_bb = b.append_block("work");
    BasicBlock* done_bb = b.append_block("done");
    b.position_at_end(entry);
    b.build_br_if(in_bounds, work_bb, done_bb);

    b.position_at_end(work_bb);
    Value* num_pairs = b.build_lshr(K, kb.const_i32(6));  // K / 64
    Value* row_base = kb.add(W, b.build_mul(b.build_sext_i64(row), b.build_sext_i64(rowbytes)));

    Value* sum = kb.for_range_reduce(
        lane, num_pairs, kb.const_i32(64), kb.const_f32(0.0f),
        [&](Value* p, Value* acc) -> Value* {
            Value* w_blk = kb.add(row_base, b.build_sext_i64(kb.mul(p, kb.const_i32(68))));
            Value* x0_ptr = kb.add(x_q8, b.build_sext_i64(kb.mul(p, kb.const_i32(72))));
            Value* x1_ptr = kb.add(x0_ptr, kb.const_i64(36));

            // Issue all 17 aligned i32 W loads and 18 X loads before arithmetic.
            Value* w_raw[17];
            for (int i = 0; i < 17; ++i) w_raw[i] = kb.load_i32(w_blk, i * 4);
            Value* dx0 = kb.load_f32(x0_ptr, 0);
            Value* xq0[8];
            for (int i = 0; i < 8; ++i) xq0[i] = kb.load_i32(x0_ptr, 4 + i * 4);
            Value* dx1 = kb.load_f32(x1_ptr, 0);
            Value* xq1[8];
            for (int i = 0; i < 8; ++i) xq1[i] = kb.load_i32(x1_ptr, 4 + i * 4);

            Value* dw0 = kb.f16_to_f32(w_raw[0]);
            Value* dw1 = kb.f16_hi_to_f32(w_raw[8]);

            Value* id0 = kb.const_i32(0);
            Value* c16 = kb.const_i32(16);
            for (int i = 0; i < 8; ++i) {
                Value* qw0 = b.build_or(b.build_lshr(w_raw[i], c16), b.build_shl(w_raw[i + 1], c16));
                id0 = kb.sdot4(qw0, xq0[i], id0);
            }
            Value* id1 = kb.const_i32(0);
            for (int i = 0; i < 8; ++i) {
                id1 = kb.sdot4(w_raw[9 + i], xq1[i], id1);
            }

            Value* s0 = kb.fma(kb.mul(dw0, dx0), kb.i32_to_f32(id0), acc);
            return kb.fma(kb.mul(dw1, dx1), kb.i32_to_f32(id1), s0);
        });

    sum = b.build_call("spirv_subgroup_add_f32", Type::f32(), {sum});

    kb.if_then(b.build_eq(lane, kb.const_i32(0)), [&]() {
        Value* y_ptr = kb.add(y, b.build_shl(b.build_sext_i64(row), kb.const_i64(1)));
        kb.store_u16(y_ptr, kb.f32_to_f16(sum));
    });

    b.build_br(done_bb);
    b.position_at_end(done_bb);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    brass::GpuOptOptions gpu_opts;
    brass::optimize_gpu_function(*fn, mod, gpu_opts);

    SpirvOptions spv_opts;
    spv_opts.stage = SpirvStage::Compute;
    spv_opts.local_size_x = kWg;
    spv_opts.local_size_spec_constants = true;
    return SpirvTarget::compile(*fn, spv_opts);
}

void run_coopmat_proof(dv::DeviceCtx& ctx, brotensor::Device vk) {
    std::printf("=== 1. Brass SPIR-V Cooperative-Matrix FP16 NT GEMM (128x128x32) ===\n");
    const SpirvKernel sk = build_coopmat_gemm_nt_f16();
    const dv::JitPipeline pipe = dv::jit_pipeline(ctx, sk, 128, /*req_subgroup=*/32);
    std::printf("  compiled %zu SPIR-V words, shared=%u B, subgroup=%u\n",
                sk.words.size(), sk.shared_bytes, pipe.subgroup);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (int sz : {512, 1024, 2048, 4096}) {
        const int M = sz, N = sz, K = sz;
        std::vector<std::uint16_t> hA(static_cast<std::size_t>(M) * K);
        std::vector<std::uint16_t> hB(static_cast<std::size_t>(N) * K);
        for (auto& v : hA) v = f32_to_f16_bits(dist(rng));
        for (auto& v : hB) v = f32_to_f16_bits(dist(rng));

        brotensor::Tensor A = brotensor::Tensor::empty_on(vk, M, K, brotensor::Dtype::FP16);
        brotensor::Tensor B = brotensor::Tensor::empty_on(vk, N, K, brotensor::Dtype::FP16);
        brotensor::Tensor C_brass = brotensor::Tensor::empty_on(vk, M, N, brotensor::Dtype::FP16);
        brotensor::Tensor C_ref = brotensor::Tensor::empty_on(vk, M, N, brotensor::Dtype::FP16);

        std::string err;
        brotensor::interop::copy_bytes_to_tensor(hA.data(), A.bytes(), A, &err);
        brotensor::interop::copy_bytes_to_tensor(hB.data(), B.bytes(), B, &err);

        const std::uint64_t a_bda = brotensor::interop::view(A).device_address;
        const std::uint64_t b_bda = brotensor::interop::view(B).device_address;
        const std::uint64_t c_bda = brotensor::interop::view(C_brass).device_address;

        std::vector<std::uint8_t> push(sk.push_constant_bytes, 0);
        set_push(push, sk, 0, a_bda);
        set_push(push, sk, 1, b_bda);
        set_push(push, sk, 2, c_bda);
        set_push(push, sk, 3, static_cast<std::int32_t>(M));
        set_push(push, sk, 4, static_cast<std::int32_t>(N));
        set_push(push, sk, 5, static_cast<std::int32_t>(K));

        const std::uint32_t gx = static_cast<std::uint32_t>(N / 128);
        const std::uint32_t gy = static_cast<std::uint32_t>(M / 128);

        brotensor::linear_forward_batched_ex(B, nullptr, A, 0, brotensor::kLinearEpiStore, nullptr, C_ref);
        dv::jit_dispatch(ctx, pipe, push.data(), gx, gy, 1);
        brotensor::sync(vk);

        std::vector<std::uint16_t> out_brass(static_cast<std::size_t>(M) * N);
        std::vector<std::uint16_t> out_ref(static_cast<std::size_t>(M) * N);
        brotensor::interop::copy_bytes_from_tensor(C_brass, out_brass.data(), C_brass.bytes(), &err);
        brotensor::interop::copy_bytes_from_tensor(C_ref, out_ref.data(), C_ref.bytes(), &err);

        float max_abs = 0.0f;
        for (std::size_t i = 0; i < out_brass.size(); ++i) {
            max_abs = std::max(max_abs, std::fabs(f16_bits_to_f32(out_brass[i]) - f16_bits_to_f32(out_ref[i])));
        }
        if (!(max_abs < 5e-2f)) {
            std::fprintf(stderr, "FAIL coopmat M=N=K=%d max_abs=%g\n", sz, max_abs);
            std::abort();
        }

        const int iters = (sz <= 1024) ? 100 : 50;
        for (int w = 0; w < 10; ++w) {
            dv::jit_dispatch(ctx, pipe, push.data(), gx, gy, 1);
        }
        brotensor::sync(vk);

        auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            dv::jit_dispatch(ctx, pipe, push.data(), gx, gy, 1);
        }
        brotensor::sync(vk);
        auto t1 = std::chrono::steady_clock::now();
        const double ms_brass = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        for (int w = 0; w < 10; ++w) {
            brotensor::linear_forward_batched_ex(B, nullptr, A, 0, brotensor::kLinearEpiStore, nullptr, C_ref);
        }
        brotensor::sync(vk);

        auto r0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            brotensor::linear_forward_batched_ex(B, nullptr, A, 0, brotensor::kLinearEpiStore, nullptr, C_ref);
        }
        brotensor::sync(vk);
        auto r1 = std::chrono::steady_clock::now();
        const double ms_ref = std::chrono::duration<double, std::milli>(r1 - r0).count() / iters;

        const double flops = 2.0 * double(M) * double(N) * double(K);
        const double tflops_brass = (flops / (ms_brass * 1e-3)) / 1e12;
        const double tflops_ref = (flops / (ms_ref * 1e-3)) / 1e12;
        std::printf("  %4dx%4dx%4d  max_abs=%.4f  brass=%.3f ms (%.2f TF/s)  gemm_cm=%.3f ms (%.2f TF/s)  ratio=%.3fx\n",
                    M, N, K, max_abs, ms_brass, tflops_brass, ms_ref, tflops_ref, tflops_brass / tflops_ref);
    }
}

void run_q8_gemv_proof(dv::DeviceCtx& ctx, brotensor::Device vk) {
    std::printf("\n=== 2. Brass SPIR-V sdot4 Q8_0 GEMV (Wave64) ===\n");
    const SpirvKernel sk_q = build_quant_x_q8();
    const SpirvKernel sk_gemv = build_gemv_q8_sdot4();
    const dv::JitPipeline pipe_q = dv::jit_pipeline(ctx, sk_q, 32, /*req_subgroup=*/32);
    const dv::JitPipeline pipe_gemv = dv::jit_pipeline(ctx, sk_gemv, 256, /*req_subgroup=*/64);
    std::printf("  compiled quant_x=%zu words (sg=%u), gemv_sdot4=%zu words (sg=%u)\n",
                sk_q.words.size(), pipe_q.subgroup, sk_gemv.words.size(), pipe_gemv.subgroup);

    struct Shape { int N, K; };
    for (Shape s : {Shape{4096, 4096}, Shape{11008, 4096}}) {
        const int N = s.N, K = s.K;
        const int nblk = K / 32;
        const int rowbytes = nblk * 34;

        std::mt19937 rng(123);
        std::uniform_real_distribution<float> scale_d(0.01f, 0.1f);
        std::uniform_int_distribution<int> q_d(-120, 120);
        std::uniform_real_distribution<float> x_d(-1.0f, 1.0f);

        std::vector<std::uint8_t> hW(static_cast<std::size_t>(N) * rowbytes);
        for (int r = 0; r < N; ++r) {
            for (int b = 0; b < nblk; ++b) {
                std::uint8_t* blk = hW.data() + static_cast<std::size_t>(r) * rowbytes + b * 34;
                const std::uint16_t d16 = f32_to_f16_bits(scale_d(rng) / 127.0f);
                std::memcpy(blk, &d16, 2);
                for (int e = 0; e < 32; ++e) {
                    blk[2 + e] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q_d(rng)));
                }
            }
        }
        std::vector<std::uint16_t> hx(K);
        for (int e = 0; e < K; ++e) hx[e] = f32_to_f16_bits(x_d(rng));

        // Exact CPU Q8_0 x FP16 & Q8_0 x Q8_1 references for row checks.
        std::vector<double> cpu_fp16(N, 0.0), cpu_q8x8(N, 0.0);
        for (int r = 0; r < N; ++r) {
            double s_fp = 0.0, s_qq = 0.0;
            for (int b = 0; b < nblk; ++b) {
                const std::uint8_t* blk = hW.data() + static_cast<std::size_t>(r) * rowbytes + b * 34;
                std::uint16_t dw16 = 0;
                std::memcpy(&dw16, blk, 2);
                const double dw = f16_bits_to_f32(dw16);
                float amax = 0.0f;
                float xf[32];
                for (int e = 0; e < 32; ++e) {
                    xf[e] = f16_bits_to_f32(hx[b * 32 + e]);
                    amax = std::max(amax, std::fabs(xf[e]));
                }
                const float dx = amax / 127.0f;
                const float inv_dx = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                std::int32_t idot = 0;
                for (int e = 0; e < 32; ++e) {
                    const auto wq = static_cast<std::int8_t>(blk[2 + e]);
                    s_fp += dw * double(wq) * double(xf[e]);
                    const float sc = std::clamp(xf[e] * inv_dx, -127.0f, 127.0f);
                    const std::int32_t xq = static_cast<std::int32_t>(sc + (sc >= 0.0f ? 0.5f : -0.5f));
                    idot += static_cast<std::int32_t>(wq) * xq;
                }
                s_qq += dw * double(dx) * double(idot);
            }
            cpu_fp16[r] = s_fp;
            cpu_q8x8[r] = s_qq;
        }

        brotensor::Tensor x_dev = brotensor::Tensor::empty_on(vk, K, 1, brotensor::Dtype::FP16);
        brotensor::Tensor x_q8 = brotensor::Tensor::empty_on(vk, nblk, 9, brotensor::Dtype::INT32);
        brotensor::Tensor W_dev = brotensor::Tensor::empty_on(vk, N, K, brotensor::Dtype::Q8_0);
        brotensor::Tensor y_brass = brotensor::Tensor::empty_on(vk, N, 1, brotensor::Dtype::FP16);
        brotensor::Tensor y_ref = brotensor::Tensor::empty_on(vk, N, 1, brotensor::Dtype::FP16);

        std::string err;
        brotensor::interop::copy_bytes_to_tensor(hx.data(), x_dev.bytes(), x_dev, &err);
        brotensor::interop::copy_bytes_to_tensor(hW.data(), hW.size(), W_dev, &err);

        const std::uint64_t x_bda = brotensor::interop::view(x_dev).device_address;
        const std::uint64_t xq_bda = brotensor::interop::view(x_q8).device_address;
        const std::uint64_t w_bda = brotensor::interop::view(W_dev).device_address;
        const std::uint64_t yb_bda = brotensor::interop::view(y_brass).device_address;

        std::vector<std::uint8_t> push_q(sk_q.push_constant_bytes, 0);
        set_push(push_q, sk_q, 0, x_bda);
        set_push(push_q, sk_q, 1, xq_bda);

        std::vector<std::uint8_t> push_g(sk_gemv.push_constant_bytes, 0);
        set_push(push_g, sk_gemv, 0, xq_bda);
        set_push(push_g, sk_gemv, 1, w_bda);
        set_push(push_g, sk_gemv, 2, yb_bda);
        set_push(push_g, sk_gemv, 3, static_cast<std::int32_t>(N));
        set_push(push_g, sk_gemv, 4, static_cast<std::int32_t>(K));
        set_push(push_g, sk_gemv, 5, static_cast<std::int32_t>(rowbytes));

        const std::uint32_t gx_gemv = static_cast<std::uint32_t>((N + 3) / 4);
        auto run_brass = [&]() {
            dv::jit_dispatch(ctx, pipe_q, push_q.data(), static_cast<std::uint32_t>(nblk), 1, 1);
            dv::jit_dispatch(ctx, pipe_gemv, push_g.data(), gx_gemv, 1, 1);
        };

        brotensor::linear_forward_q8_0_fp16(W_dev, nullptr, x_dev, y_ref);
        run_brass();
        brotensor::sync(vk);

        std::vector<std::uint16_t> hy_b(N), hy_r(N);
        brotensor::interop::copy_bytes_from_tensor(y_brass, hy_b.data(), y_brass.bytes(), &err);
        brotensor::interop::copy_bytes_from_tensor(y_ref, hy_r.data(), y_ref.bytes(), &err);

        double dot = 0.0, nb = 0.0, nr = 0.0;
        double diff_qq_sq = 0.0, norm_qq_sq = 0.0;
        double max_abs_ref = 0.0, max_abs_qq = 0.0;
        for (int r = 0; r < N; ++r) {
            const double vb = f16_bits_to_f32(hy_b[r]);
            const double vr = f16_bits_to_f32(hy_r[r]);
            const double vq = double(f16_bits_to_f32(f32_to_f16_bits(static_cast<float>(cpu_q8x8[r]))));
            dot += vb * vr;
            nb += vb * vb;
            nr += vr * vr;
            diff_qq_sq += (vb - vq) * (vb - vq);
            norm_qq_sq += vq * vq;
            max_abs_ref = std::max(max_abs_ref, std::fabs(vb - vr));
            max_abs_qq = std::max(max_abs_qq, std::fabs(vb - vq));
        }
        const double cos_sim = dot / (std::sqrt(nb) * std::sqrt(nr));
        const double l2_rel_qq = std::sqrt(diff_qq_sq / std::max(norm_qq_sq, 1e-12));
        if (!(cos_sim > 0.9999 && l2_rel_qq < 1e-3)) {
            std::fprintf(stderr, "FAIL q8 gemv N=%d K=%d cos=%.6f l2_rel_qq=%g max_abs_qq=%g max_abs_ref=%g\n",
                         N, K, cos_sim, l2_rel_qq, max_abs_qq, max_abs_ref);
            std::abort();
        }

        // L2-defeating rotation over 512 MiB of distinct W tensors (matching bench_quant_matmul).
        const std::size_t bytes_w = hW.size();
        const int copies = std::max(1, static_cast<int>((512ULL << 20) / bytes_w));
        std::vector<brotensor::Tensor> W_ring;
        std::vector<std::uint64_t> w_bdas;
        W_ring.reserve(static_cast<std::size_t>(copies));
        w_bdas.reserve(static_cast<std::size_t>(copies));
        W_ring.push_back(std::move(W_dev));
        w_bdas.push_back(w_bda);
        for (int c = 1; c < copies; ++c) {
            brotensor::Tensor Wc = brotensor::Tensor::empty_on(vk, N, K, brotensor::Dtype::Q8_0);
            brotensor::interop::copy_bytes_to_tensor(hW.data(), bytes_w, Wc, &err);
            w_bdas.push_back(brotensor::interop::view(Wc).device_address);
            W_ring.push_back(std::move(Wc));
        }
        brotensor::sync(vk);

        const int iters = 200;
        for (int w = 0; w < 20; ++w) {
            set_push(push_g, sk_gemv, 1, w_bdas[static_cast<std::size_t>(w % copies)]);
            run_brass();
        }
        brotensor::sync(vk);
        auto b0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            set_push(push_g, sk_gemv, 1, w_bdas[static_cast<std::size_t>(it % copies)]);
            run_brass();
        }
        brotensor::sync(vk);
        auto b1 = std::chrono::steady_clock::now();
        const double ms_brass = std::chrono::duration<double, std::milli>(b1 - b0).count() / iters;

        for (int w = 0; w < 20; ++w) {
            brotensor::linear_forward_q8_0_fp16(W_ring[static_cast<std::size_t>(w % copies)], nullptr, x_dev, y_ref);
        }
        brotensor::sync(vk);
        auto g0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            brotensor::linear_forward_q8_0_fp16(W_ring[static_cast<std::size_t>(it % copies)], nullptr, x_dev, y_ref);
        }
        brotensor::sync(vk);
        auto g1 = std::chrono::steady_clock::now();
        const double ms_ref = std::chrono::duration<double, std::milli>(g1 - g0).count() / iters;

        const double gb = double(bytes_w) / 1e9;
        const double gbps_brass = gb / (ms_brass * 1e-3);
        const double gbps_ref = gb / (ms_ref * 1e-3);
        std::printf("  %5dx%-4d  cos=%.6f l2_rel_qq=%.2e max_abs_qq=%.4f  brass=%.3f ms (%.1f GB/s)  gemv_q=%.3f ms (%.1f GB/s)  ratio=%.3fx\n",
                    N, K, cos_sim, l2_rel_qq, max_abs_qq, ms_brass, gbps_brass, ms_ref, gbps_ref, gbps_brass / gbps_ref);
    }
}

}  // namespace

int main() {
    brotensor::init();
    const brotensor::Device vk = brotensor::Device::vulkan(0);
    if (!brotensor::is_available(vk)) {
        std::printf("No Vulkan device available\n");
        return 1;
    }
    dv::DeviceCtx& ctx = dv::device(0);
    std::printf("Device: %s (driver: %s, coopmat_f16=%d, int_dot=%d, sg=%u..%u)\n",
                ctx.info().name.c_str(), ctx.info().driver.c_str(),
                int(ctx.info().coopmat_f16), int(ctx.info().integer_dot_product),
                ctx.info().min_subgroup, ctx.info().max_subgroup);

    run_coopmat_proof(ctx, vk);
    run_q8_gemv_proof(ctx, vk);
    brotensor::shutdown();
    return 0;
}
