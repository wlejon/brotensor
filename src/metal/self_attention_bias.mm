// ─── Metal self-attention with additive pre-softmax bias ───────────────────
//
// Metal counterpart of src/cuda/self_attention_bias.cu — multi-head
// self-attention that adds an optional per-head (L, L) bias to the attention
// logits before softmax (the primitive behind T5 relative-position bias and
// ALiBi-style biases).
//
//   S[h,q,k] = scale * (Q_h[q] . K_h[k]) + attn_bias[h*L+q, k]
//   O        = concat_h( softmax_k(S[h]) @ V_h ) @ Wo
//
// Scores are materialised (L, L) per head — intended for encoder-length
// sequences, not long-context decoding. Dispatched on X.dtype (FP32 / FP16 /
// BF16): the projection inputs/outputs are typed, every intermediate
// (Q/K/V/scores/softmax) is FP32 scratch, math is FP32. attn_bias is FP32.
//
// Also here: `rel_pos_bias_xl_forward`, the Transformer-XL relative-position
// *producer* of that bias (Conformer / FastConformer / Transformer-XL).

#include <brotensor/runtime.h>

#include <cmath>
#include <stdexcept>

#import "internal.h"
#import "self_attention_bias_msl.h"

namespace brotensor::detail::metal {

using metal_impl::buffer_for;
using metal_impl::buffer_offset_for;
using metal_impl::compile_pipeline;
using metal_impl::new_command_buffer;
using metal_impl::pool_require;

namespace {

constexpr NSUInteger kSabSoftmaxBlock = 256;

// Threadgroup staging area for one query slice in k_rel_pos_bias_xl — must
// match RPB_XL_MAX_HEAD_DIM in the MSL source below.
constexpr int kRelPosXlMaxHeadDim = 512;

#define DEF_PSO(NAME, FN) \
    id<MTLComputePipelineState> NAME() { \
        static dispatch_once_t once; \
        static id<MTLComputePipelineState> pso; \
        dispatch_once(&once, ^{ pso = compile_pipeline(kSrc, FN); }); \
        return pso; \
    }
DEF_PSO(pso_proj_fp32,   @"k_sab_proj_fp32")
DEF_PSO(pso_proj_fp16,   @"k_sab_proj_fp16")
DEF_PSO(pso_proj_bf16,   @"k_sab_proj_bf16")
DEF_PSO(pso_scores,      @"k_sab_scores")
DEF_PSO(pso_softmax,     @"k_sab_softmax")
DEF_PSO(pso_apply_v,     @"k_sab_apply_v")
DEF_PSO(pso_output_fp32, @"k_sab_output_fp32")
DEF_PSO(pso_output_fp16, @"k_sab_output_fp16")
DEF_PSO(pso_output_bf16, @"k_sab_output_bf16")
DEF_PSO(pso_proj_int8,   @"k_sab_proj_int8")
DEF_PSO(pso_output_int8, @"k_sab_output_int8")
DEF_PSO(pso_sardp_proj_fp32,    @"k_sardp_proj_fp32")
DEF_PSO(pso_sardp_proj_fp16,    @"k_sardp_proj_fp16")
DEF_PSO(pso_sardp_proj_bf16,    @"k_sardp_proj_bf16")
DEF_PSO(pso_sardp_scores_fp32,  @"k_sardp_scores_fp32")
DEF_PSO(pso_sardp_scores_fp16,  @"k_sardp_scores_fp16")
DEF_PSO(pso_sardp_scores_bf16,  @"k_sardp_scores_bf16")
DEF_PSO(pso_sardp_output_fp32,  @"k_sardp_output_fp32")
DEF_PSO(pso_sardp_output_fp16,  @"k_sardp_output_fp16")
DEF_PSO(pso_sardp_output_bf16,  @"k_sardp_output_bf16")
DEF_PSO(pso_win_gather_fp32,    @"k_sardp_win_gather_fp32")
DEF_PSO(pso_win_gather_fp16,    @"k_sardp_win_gather_fp16")
DEF_PSO(pso_win_gather_bf16,    @"k_sardp_win_gather_bf16")
DEF_PSO(pso_win_scatter_fp32,   @"k_sardp_win_scatter_fp32")
DEF_PSO(pso_win_scatter_fp16,   @"k_sardp_win_scatter_fp16")
DEF_PSO(pso_win_scatter_bf16,   @"k_sardp_win_scatter_bf16")
DEF_PSO(pso_rel_pos_bias_xl,    @"k_rel_pos_bias_xl")
#undef DEF_PSO

// Forward declarations — run3d / run_rows are defined just below.
void run3d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny,
           NSUInteger nz, void (^bind)(id<MTLComputeCommandEncoder>));
void run_rows(id<MTLComputePipelineState> pso, NSUInteger rows,
              void (^bind)(id<MTLComputeCommandEncoder>));

// 2-D thread grid (nx, ny), threadgroup capped at 16x16. For win gather/scatter.
void run2d(id<MTLComputePipelineState> pso, NSUInteger nx, NSUInteger ny,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tx = nx < 16 ? nx : 16;
        NSUInteger ty = ny < 16 ? ny : 16;
        [enc dispatchThreads:MTLSizeMake(nx, ny, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// Core decomposed-rel-pos attention over one token grid (gh x gw = L rows).
// X / O are passed as buffer + byte offset so the windowed path can drive
// per-window slices of a partition buffer. Wq/Wk/Wv/Wo and rel_h/rel_w are
// whole tensors; biases are optional. dt selects the typed proj/scores/output
// kernels; every intermediate is FP32 scratch.
void run_sardp(Dtype dt,
               id<MTLBuffer> bX, NSUInteger oX,
               const Tensor& Wq, const Tensor* bq,
               const Tensor& Wk, const Tensor* bk,
               const Tensor& Wv, const Tensor* bv,
               const Tensor& Wo, const Tensor* bo,
               const Tensor& rel_h, const Tensor& rel_w,
               int H, int L, int D, int dh, int gh, int gw, float scale,
               id<MTLBuffer> bO, NSUInteger oO) {
    Tensor Qh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor S  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor A  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor Yc = Tensor::empty_on(Device::Metal, L, D, Dtype::FP32);

    id<MTLBuffer> bQh = buffer_for(Qh); NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh); NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh); NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bS  = buffer_for(S);  NSUInteger oS  = buffer_offset_for(S);
    id<MTLBuffer> bA  = buffer_for(A);  NSUInteger oA  = buffer_offset_for(A);
    id<MTLBuffer> bYc = buffer_for(Yc); NSUInteger oYc = buffer_offset_for(Yc);

    id<MTLComputePipelineState> proj_pso =
        (dt == Dtype::FP16) ? pso_sardp_proj_fp16()
      : (dt == Dtype::BF16) ? pso_sardp_proj_bf16()
      : pso_sardp_proj_fp32();
    id<MTLComputePipelineState> scores_pso =
        (dt == Dtype::FP16) ? pso_sardp_scores_fp16()
      : (dt == Dtype::BF16) ? pso_sardp_scores_bf16()
      : pso_sardp_scores_fp32();
    id<MTLComputePipelineState> out_pso =
        (dt == Dtype::FP16) ? pso_sardp_output_fp16()
      : (dt == Dtype::BF16) ? pso_sardp_output_bf16()
      : pso_sardp_output_fp32();

    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t dhu = static_cast<uint32_t>(dh);
    const uint32_t ghu = static_cast<uint32_t>(gh);
    const uint32_t gwu = static_cast<uint32_t>(gw);

    // Per-head projection with optional bias.
    auto proj = [&](const Tensor& W, const Tensor* b,
                    id<MTLBuffer> bOut, NSUInteger oOut) {
        id<MTLBuffer> bW = buffer_for(W); NSUInteger oW = buffer_offset_for(W);
        const bool hb = (b && b->data);
        id<MTLBuffer> bB = hb ? buffer_for(*b) : bW;
        NSUInteger oB = hb ? buffer_offset_for(*b) : oW;
        const uint32_t has_bias = hb ? 1u : 0u;
        run3d(proj_pso, dhu, Lu, static_cast<NSUInteger>(H),
              ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX   offset:oX   atIndex:0];
            [enc setBuffer:bW   offset:oW   atIndex:1];
            [enc setBuffer:bB   offset:oB   atIndex:2];
            [enc setBytes:&has_bias length:sizeof(uint32_t) atIndex:3];
            [enc setBuffer:bOut offset:oOut atIndex:4];
            [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:7];
        });
    };
    proj(Wq, bq, bQh, oQh);
    proj(Wk, bk, bKh, oKh);
    proj(Wv, bv, bVh, oVh);

    // Scores with inline decomposed rel-pos bias.
    id<MTLBuffer> bRh = buffer_for(rel_h); NSUInteger oRh = buffer_offset_for(rel_h);
    id<MTLBuffer> bRw = buffer_for(rel_w); NSUInteger oRw = buffer_offset_for(rel_w);
    run3d(scores_pso, Lu, Lu, static_cast<NSUInteger>(H),
          ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQh offset:oQh atIndex:0];
        [enc setBuffer:bKh offset:oKh atIndex:1];
        [enc setBuffer:bRh offset:oRh atIndex:2];
        [enc setBuffer:bRw offset:oRw atIndex:3];
        [enc setBuffer:bS  offset:oS  atIndex:4];
        [enc setBytes:&Lu    length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhu   length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&scale length:sizeof(float)    atIndex:7];
        [enc setBytes:&ghu   length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&gwu   length:sizeof(uint32_t) atIndex:9];
    });

    // Softmax (no mask) — reuse the bias-path row softmax.
    run_rows(pso_softmax(), static_cast<NSUInteger>(H) * L,
             ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS offset:oS atIndex:0];
        [enc setBuffer:bA offset:oA atIndex:1];
        [enc setBuffer:bS offset:oS atIndex:2];  // dummy mask
        const uint32_t has_mask = 0u;
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lu       length:sizeof(uint32_t) atIndex:4];
    });

    // Attn @ V → Yconcat — reuse the bias-path apply-V kernel.
    run3d(pso_apply_v(), dhu, Lu, static_cast<NSUInteger>(H),
          ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA  offset:oA  atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bYc offset:oYc atIndex:2];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:5];
    });

    // Output projection with optional bias.
    id<MTLBuffer> bWo = buffer_for(Wo); NSUInteger oWo = buffer_offset_for(Wo);
    const bool hbo = (bo && bo->data);
    id<MTLBuffer> bBo = hbo ? buffer_for(*bo) : bWo;
    NSUInteger oBo = hbo ? buffer_offset_for(*bo) : oWo;
    const uint32_t has_bo = hbo ? 1u : 0u;
    run3d(out_pso, Du, Lu, 1, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bYc offset:oYc atIndex:0];
        [enc setBuffer:bWo offset:oWo atIndex:1];
        [enc setBuffer:bBo offset:oBo atIndex:2];
        [enc setBytes:&has_bo length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bO  offset:oO  atIndex:4];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:6];
    });
}

// Dispatch a 3-D thread grid (nx, ny, nz), threadgroup capped at 256 threads.
void run3d(id<MTLComputePipelineState> pso,
           NSUInteger nx, NSUInteger ny, NSUInteger nz,
           void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (nx == 0 || ny == 0 || nz == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        NSUInteger tx = nx < 16 ? nx : 16;
        NSUInteger ty = ny < 16 ? ny : 16;
        [enc dispatchThreads:MTLSizeMake(nx, ny, nz)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

// One threadgroup per row; `kSabSoftmaxBlock` threads each.
void run_rows(id<MTLComputePipelineState> pso, NSUInteger rows,
              void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (rows == 0) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        bind(enc);
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kSabSoftmaxBlock, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // namespace

void self_attention_bias_forward(const Tensor& X, const Tensor& Wq,
                                 const Tensor& Wk, const Tensor& Wv,
                                 const Tensor& Wo,
                                 const Tensor* bq, const Tensor* bk,
                                 const Tensor* bv, const Tensor* bo,
                                 const float* d_mask,
                                 const Tensor* attn_bias, int num_heads,
                                 float scale, Tensor& O) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16) {
        throw std::runtime_error("self_attention_bias_forward: X must be FP32, FP16, or BF16");
    }
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo dtype must match X");
    }
    const int L = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    if (H <= 0 || D % H != 0) {
        throw std::runtime_error("self_attention_bias_forward: num_heads must divide D");
    }
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D) {
        throw std::runtime_error("self_attention_bias_forward: Wq/Wk/Wv/Wo must be (D, D)");
    }
    const int dh = D / H;

    bool has_bias = false;
    if (attn_bias && attn_bias->data) {
        if (attn_bias->dtype != Dtype::FP32) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be FP32");
        }
        if (attn_bias->size() != H * L * L) {
            throw std::runtime_error("self_attention_bias_forward: attn_bias must be (num_heads*L, L)");
        }
        has_bias = true;
    }
    if (O.rows != L || O.cols != D || O.dtype != X.dtype) {
        O.resize(L, D, X.dtype);
    }
    if (L == 0 || D == 0) return;

    // FP32 scratch for every intermediate.
    Tensor Qh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor S  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor A  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor Yc = Tensor::empty_on(Device::Metal, L, D, Dtype::FP32);

    id<MTLBuffer> bX = buffer_for(X);   NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bO = buffer_for(O);   NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bQh = buffer_for(Qh); NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh); NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh); NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bS  = buffer_for(S);  NSUInteger oS  = buffer_offset_for(S);
    id<MTLBuffer> bA  = buffer_for(A);  NSUInteger oA  = buffer_offset_for(A);
    id<MTLBuffer> bYc = buffer_for(Yc); NSUInteger oYc = buffer_offset_for(Yc);

    NSUInteger oM = 0;
    id<MTLBuffer> bM = pool_require(d_mask, oM, "self_attention_bias_forward", "mask");
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;
    const uint32_t has_mask = (bM != nil) ? 1u : 0u;

    id<MTLBuffer> bB = has_bias ? buffer_for(*attn_bias) : nil;
    NSUInteger oB = has_bias ? buffer_offset_for(*attn_bias) : 0;
    id<MTLBuffer> bB_arg = bB ? bB : bX;
    NSUInteger oB_arg = bB ? oB : oX;
    const uint32_t has_bias_u = has_bias ? 1u : 0u;

    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t dhu = static_cast<uint32_t>(dh);

    id<MTLComputePipelineState> proj_pso =
        (X.dtype == Dtype::FP16) ? pso_proj_fp16()
      : (X.dtype == Dtype::BF16) ? pso_proj_bf16()
      : pso_proj_fp32();

    // Q / K / V per-head projections, each with an optional bias.
    auto proj = [&](const Tensor& W, const Tensor* b,
                    id<MTLBuffer> bOut, NSUInteger oOut) {
        id<MTLBuffer> bW = buffer_for(W); NSUInteger oW = buffer_offset_for(W);
        const bool hb = (b && b->data);
        id<MTLBuffer> bB = hb ? buffer_for(*b) : bW;
        NSUInteger oB = hb ? buffer_offset_for(*b) : oW;
        const uint32_t hb_u = hb ? 1u : 0u;
        run3d(proj_pso, dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX   offset:oX   atIndex:0];
            [enc setBuffer:bW   offset:oW   atIndex:1];
            [enc setBuffer:bB   offset:oB   atIndex:2];
            [enc setBytes:&hb_u length:sizeof(uint32_t) atIndex:3];
            [enc setBuffer:bOut offset:oOut atIndex:4];
            [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:7];
        });
    };
    proj(Wq, bq, bQh, oQh);
    proj(Wk, bk, bKh, oKh);
    proj(Wv, bv, bVh, oVh);

    // Scores: scale * (Q.K) + bias.
    run3d(pso_scores(), Lu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQh   offset:oQh   atIndex:0];
        [enc setBuffer:bKh   offset:oKh   atIndex:1];
        [enc setBuffer:bB_arg offset:oB_arg atIndex:2];
        [enc setBytes:&has_bias_u length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bS    offset:oS    atIndex:4];
        [enc setBytes:&Lu    length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhu   length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&scale length:sizeof(float)    atIndex:7];
    });

    // Row-wise masked softmax.
    run_rows(pso_softmax(), static_cast<NSUInteger>(H) * L,
             ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS     offset:oS     atIndex:0];
        [enc setBuffer:bA     offset:oA     atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lu       length:sizeof(uint32_t) atIndex:4];
    });

    // Attn @ V → Yconcat.
    run3d(pso_apply_v(), dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA  offset:oA  atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bYc offset:oYc atIndex:2];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:5];
    });

    // Output projection by Wo.
    id<MTLComputePipelineState> out_pso =
        (X.dtype == Dtype::FP16) ? pso_output_fp16()
      : (X.dtype == Dtype::BF16) ? pso_output_bf16()
      : pso_output_fp32();
    id<MTLBuffer> bWo = buffer_for(Wo);
    NSUInteger oWo = buffer_offset_for(Wo);
    const bool hbo = (bo && bo->data);
    id<MTLBuffer> bBo = hbo ? buffer_for(*bo) : bWo;
    NSUInteger oBo = hbo ? buffer_offset_for(*bo) : oWo;
    const uint32_t has_bo = hbo ? 1u : 0u;
    run3d(out_pso, Du, Lu, 1, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bYc    offset:oYc    atIndex:0];
        [enc setBuffer:bWo    offset:oWo    atIndex:1];
        [enc setBuffer:bBo    offset:oBo    atIndex:2];
        [enc setBytes:&has_bo length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:4];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:5];
        [enc setBuffer:bO     offset:oO     atIndex:6];
        [enc setBytes:&Lu     length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&Du     length:sizeof(uint32_t) atIndex:8];
    });
}

// W8A16 INT8 weight-only variant. The four projection weights are INT8 (D, D)
// matrices paired with FP32 (D, 1) per-output-row dequant scales; activations
// stay FP16. The attention core (scores / softmax / Attn@V) is byte-identical
// to the FP16 path above — only the projection and output matmuls differ.
void self_attention_bias_int8w_fp16(const Tensor& X,
                                    const Tensor& Wq_int8, const Tensor& sq,
                                    const Tensor& Wk_int8, const Tensor& sk,
                                    const Tensor& Wv_int8, const Tensor& sv,
                                    const Tensor& Wo_int8, const Tensor& so,
                                    const float* d_mask,
                                    const Tensor* attn_bias, int num_heads,
                                    float scale, Tensor& O) {
    if (X.dtype != Dtype::FP16) {
        throw std::runtime_error("self_attention_bias_int8w_fp16: X must be FP16");
    }
    if (Wq_int8.dtype != Dtype::INT8 || Wk_int8.dtype != Dtype::INT8 ||
        Wv_int8.dtype != Dtype::INT8 || Wo_int8.dtype != Dtype::INT8) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: Wq/Wk/Wv/Wo must be INT8");
    }
    if (sq.dtype != Dtype::FP32 || sk.dtype != Dtype::FP32 ||
        sv.dtype != Dtype::FP32 || so.dtype != Dtype::FP32) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: scales must be FP32");
    }
    const int L = X.rows;
    const int D = X.cols;
    const int H = num_heads;
    if (H <= 0 || D % H != 0) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: num_heads must divide D");
    }
    if (Wq_int8.rows != D || Wq_int8.cols != D ||
        Wk_int8.rows != D || Wk_int8.cols != D ||
        Wv_int8.rows != D || Wv_int8.cols != D ||
        Wo_int8.rows != D || Wo_int8.cols != D) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: Wq/Wk/Wv/Wo must be (D, D)");
    }
    if (sq.size() != D || sk.size() != D || sv.size() != D || so.size() != D) {
        throw std::runtime_error(
            "self_attention_bias_int8w_fp16: each scale tensor must have D entries");
    }
    const int dh = D / H;

    bool has_bias = false;
    if (attn_bias && attn_bias->data) {
        if (attn_bias->dtype != Dtype::FP32) {
            throw std::runtime_error(
                "self_attention_bias_int8w_fp16: attn_bias must be FP32");
        }
        if (attn_bias->size() != H * L * L) {
            throw std::runtime_error(
                "self_attention_bias_int8w_fp16: attn_bias must be (num_heads*L, L)");
        }
        has_bias = true;
    }
    if (O.rows != L || O.cols != D || O.dtype != Dtype::FP16) {
        O.resize(L, D, Dtype::FP16);
    }
    if (L == 0 || D == 0) return;

    // FP32 scratch for every intermediate.
    Tensor Qh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Kh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor Vh = Tensor::empty_on(Device::Metal, H * L, dh, Dtype::FP32);
    Tensor S  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor A  = Tensor::empty_on(Device::Metal, H * L, L,  Dtype::FP32);
    Tensor Yc = Tensor::empty_on(Device::Metal, L, D, Dtype::FP32);

    id<MTLBuffer> bX = buffer_for(X);   NSUInteger oX = buffer_offset_for(X);
    id<MTLBuffer> bO = buffer_for(O);   NSUInteger oO = buffer_offset_for(O);
    id<MTLBuffer> bQh = buffer_for(Qh); NSUInteger oQh = buffer_offset_for(Qh);
    id<MTLBuffer> bKh = buffer_for(Kh); NSUInteger oKh = buffer_offset_for(Kh);
    id<MTLBuffer> bVh = buffer_for(Vh); NSUInteger oVh = buffer_offset_for(Vh);
    id<MTLBuffer> bS  = buffer_for(S);  NSUInteger oS  = buffer_offset_for(S);
    id<MTLBuffer> bA  = buffer_for(A);  NSUInteger oA  = buffer_offset_for(A);
    id<MTLBuffer> bYc = buffer_for(Yc); NSUInteger oYc = buffer_offset_for(Yc);

    NSUInteger oM = 0;
    id<MTLBuffer> bM = pool_require(d_mask, oM, "self_attention_bias_int8w_fp16", "mask");
    id<MTLBuffer> bM_arg = bM ? bM : bX;
    NSUInteger oM_arg = bM ? oM : oX;
    const uint32_t has_mask = (bM != nil) ? 1u : 0u;

    id<MTLBuffer> bB = has_bias ? buffer_for(*attn_bias) : nil;
    NSUInteger oB = has_bias ? buffer_offset_for(*attn_bias) : 0;
    id<MTLBuffer> bB_arg = bB ? bB : bX;
    NSUInteger oB_arg = bB ? oB : oX;
    const uint32_t has_bias_u = has_bias ? 1u : 0u;

    const uint32_t Lu = static_cast<uint32_t>(L);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t dhu = static_cast<uint32_t>(dh);

    // Q / K / V per-head INT8 projections.
    auto proj = ^(id<MTLBuffer> bW, NSUInteger oW,
                  id<MTLBuffer> bSc, NSUInteger oSc,
                  id<MTLBuffer> bOut, NSUInteger oOut) {
        run3d(pso_proj_int8(), dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:bX   offset:oX   atIndex:0];
            [enc setBuffer:bW   offset:oW   atIndex:1];
            [enc setBuffer:bSc  offset:oSc  atIndex:2];
            [enc setBuffer:bOut offset:oOut atIndex:3];
            [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:6];
        });
    };
    proj(buffer_for(Wq_int8), buffer_offset_for(Wq_int8),
         buffer_for(sq), buffer_offset_for(sq), bQh, oQh);
    proj(buffer_for(Wk_int8), buffer_offset_for(Wk_int8),
         buffer_for(sk), buffer_offset_for(sk), bKh, oKh);
    proj(buffer_for(Wv_int8), buffer_offset_for(Wv_int8),
         buffer_for(sv), buffer_offset_for(sv), bVh, oVh);

    // Scores: scale * (Q.K) + bias.
    run3d(pso_scores(), Lu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQh   offset:oQh   atIndex:0];
        [enc setBuffer:bKh   offset:oKh   atIndex:1];
        [enc setBuffer:bB_arg offset:oB_arg atIndex:2];
        [enc setBytes:&has_bias_u length:sizeof(uint32_t) atIndex:3];
        [enc setBuffer:bS    offset:oS    atIndex:4];
        [enc setBytes:&Lu    length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&dhu   length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&scale length:sizeof(float)    atIndex:7];
    });

    // Row-wise masked softmax.
    run_rows(pso_softmax(), static_cast<NSUInteger>(H) * L,
             ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bS     offset:oS     atIndex:0];
        [enc setBuffer:bA     offset:oA     atIndex:1];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:2];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Lu       length:sizeof(uint32_t) atIndex:4];
    });

    // Attn @ V → Yconcat.
    run3d(pso_apply_v(), dhu, Lu, H, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bA  offset:oA  atIndex:0];
        [enc setBuffer:bVh offset:oVh atIndex:1];
        [enc setBuffer:bYc offset:oYc atIndex:2];
        [enc setBytes:&Lu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dhu length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:5];
    });

    // INT8 output projection by Wo.
    id<MTLBuffer> bWo = buffer_for(Wo_int8);
    NSUInteger oWo = buffer_offset_for(Wo_int8);
    id<MTLBuffer> bSo = buffer_for(so);
    NSUInteger oSo = buffer_offset_for(so);
    run3d(pso_output_int8(), Du, Lu, 1, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bYc    offset:oYc    atIndex:0];
        [enc setBuffer:bWo    offset:oWo    atIndex:1];
        [enc setBuffer:bSo    offset:oSo    atIndex:2];
        [enc setBuffer:bM_arg offset:oM_arg atIndex:3];
        [enc setBytes:&has_mask length:sizeof(uint32_t) atIndex:4];
        [enc setBuffer:bO     offset:oO     atIndex:5];
        [enc setBytes:&Lu     length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&Du     length:sizeof(uint32_t) atIndex:7];
    });
}

// ── Decomposed 2D relative-position attention (public entry points) ─────────

namespace {

void check_rdp_common(const char* fn, const Tensor& X,
                      const Tensor& Wq, const Tensor& Wk,
                      const Tensor& Wv, const Tensor& Wo,
                      const Tensor& rel_h, const Tensor& rel_w,
                      const Tensor* bq, const Tensor* bk,
                      const Tensor* bv, const Tensor* bo,
                      int num_heads, int D) {
    if (X.dtype != Dtype::FP32 && X.dtype != Dtype::FP16 && X.dtype != Dtype::BF16)
        throw std::runtime_error(std::string(fn) + ": X must be FP32, FP16, or BF16");
    if (Wq.dtype != X.dtype || Wk.dtype != X.dtype ||
        Wv.dtype != X.dtype || Wo.dtype != X.dtype ||
        rel_h.dtype != X.dtype || rel_w.dtype != X.dtype)
        throw std::runtime_error(std::string(fn) +
            ": Wq/Wk/Wv/Wo/rel_pos_h/rel_pos_w dtype must match X");
    if (num_heads <= 0 || D % num_heads != 0)
        throw std::runtime_error(std::string(fn) + ": num_heads must divide D");
    if (Wq.rows != D || Wq.cols != D || Wk.rows != D || Wk.cols != D ||
        Wv.rows != D || Wv.cols != D || Wo.rows != D || Wo.cols != D)
        throw std::runtime_error(std::string(fn) + ": Wq/Wk/Wv/Wo must be (D, D)");
    auto check_bias = [&](const Tensor* b, const char* name) {
        if (b && b->data) {
            if (b->dtype != X.dtype)
                throw std::runtime_error(std::string(fn) + ": " + name + " dtype must match X");
            if (b->size() != D)
                throw std::runtime_error(std::string(fn) + ": " + name + " must have D entries");
        }
    };
    check_bias(bq, "bq"); check_bias(bk, "bk");
    check_bias(bv, "bv"); check_bias(bo, "bo");
}

} // namespace

void self_attention_decomposed_rel_pos_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, float scale, Tensor& O) {
    const char* fn = "self_attention_decomposed_rel_pos_forward";
    const int L = X.rows;
    const int D = X.cols;
    check_rdp_common(fn, X, Wq, Wk, Wv, Wo, rel_pos_h, rel_pos_w,
                     bq, bk, bv, bo, num_heads, D);
    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != L)
        throw std::runtime_error(std::string(fn) + ": grid_h*grid_w must equal X.rows");
    const int dh = D / num_heads;
    if (rel_pos_h.rows != 2 * grid_h - 1 || rel_pos_h.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_h must be (2*grid_h-1, head_dim)");
    if (rel_pos_w.rows != 2 * grid_w - 1 || rel_pos_w.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_w must be (2*grid_w-1, head_dim)");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype)
        O.resize(L, D, X.dtype);
    if (L == 0 || D == 0) return;

    run_sardp(X.dtype, buffer_for(X), buffer_offset_for(X),
              Wq, bq, Wk, bk, Wv, bv, Wo, bo, rel_pos_h, rel_pos_w,
              num_heads, L, D, dh, grid_h, grid_w, scale,
              buffer_for(O), buffer_offset_for(O));
}

void self_attention_decomposed_rel_pos_windowed_forward(
        const Tensor& X,
        const Tensor& Wq, const Tensor* bq,
        const Tensor& Wk, const Tensor* bk,
        const Tensor& Wv, const Tensor* bv,
        const Tensor& Wo, const Tensor* bo,
        const Tensor& rel_pos_h, const Tensor& rel_pos_w,
        int num_heads, int grid_h, int grid_w, int window, float scale,
        Tensor& O) {
    const char* fn = "self_attention_decomposed_rel_pos_windowed_forward";
    const int L = X.rows;
    const int D = X.cols;
    check_rdp_common(fn, X, Wq, Wk, Wv, Wo, rel_pos_h, rel_pos_w,
                     bq, bk, bv, bo, num_heads, D);
    if (window <= 0)
        throw std::runtime_error(std::string(fn) + ": window must be >= 1");
    if (grid_h <= 0 || grid_w <= 0 || grid_h * grid_w != L)
        throw std::runtime_error(std::string(fn) + ": grid_h*grid_w must equal X.rows");
    const int dh = D / num_heads;
    if (rel_pos_h.rows != 2 * window - 1 || rel_pos_h.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_h must be (2*window-1, head_dim)");
    if (rel_pos_w.rows != 2 * window - 1 || rel_pos_w.cols != dh)
        throw std::runtime_error(std::string(fn) + ": rel_pos_w must be (2*window-1, head_dim)");
    if (O.rows != L || O.cols != D || O.dtype != X.dtype)
        O.resize(L, D, X.dtype);
    if (L == 0 || D == 0) return;

    const Dtype dt = X.dtype;
    const int pad_h = (window - grid_h % window) % window;
    const int pad_w = (window - grid_w % window) % window;
    const int nw_h  = (grid_h + pad_h) / window;
    const int nw_w  = (grid_w + pad_w) / window;
    const int nW    = nw_h * nw_w;
    const int ww    = window * window;
    const int nrows = nW * ww;

    Tensor Pin  = Tensor::empty_on(Device::Metal, nrows, D, dt);
    Tensor Pout = Tensor::empty_on(Device::Metal, nrows, D, dt);
    id<MTLBuffer> bPin  = buffer_for(Pin);  NSUInteger oPin  = buffer_offset_for(Pin);
    id<MTLBuffer> bPout = buffer_for(Pout); NSUInteger oPout = buffer_offset_for(Pout);

    id<MTLComputePipelineState> gather_pso =
        (dt == Dtype::FP16) ? pso_win_gather_fp16()
      : (dt == Dtype::BF16) ? pso_win_gather_bf16()
      : pso_win_gather_fp32();
    id<MTLComputePipelineState> scatter_pso =
        (dt == Dtype::FP16) ? pso_win_scatter_fp16()
      : (dt == Dtype::BF16) ? pso_win_scatter_bf16()
      : pso_win_scatter_fp32();

    const uint32_t ghu = static_cast<uint32_t>(grid_h);
    const uint32_t gwu = static_cast<uint32_t>(grid_w);
    const uint32_t winu = static_cast<uint32_t>(window);
    const uint32_t nwwu = static_cast<uint32_t>(nw_w);
    const uint32_t Du = static_cast<uint32_t>(D);
    const uint32_t nrowsu = static_cast<uint32_t>(nrows);

    auto win_args = ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBytes:&ghu    length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&gwu    length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&winu   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&nwwu   length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&Du     length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&nrowsu length:sizeof(uint32_t) atIndex:7];
    };

    // Gather grid → padded per-window partition.
    run2d(gather_pso, Du, nrowsu, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:buffer_for(X) offset:buffer_offset_for(X) atIndex:0];
        [enc setBuffer:bPin offset:oPin atIndex:1];
        win_args(enc);
    });

    // Per-window attention over contiguous row-slices of the partition buffers.
    const NSUInteger row_stride =
        static_cast<NSUInteger>(ww) * D * ::brotensor::dtype_size_bytes(dt);
    for (int w = 0; w < nW; ++w) {
        const NSUInteger inOff  = oPin  + static_cast<NSUInteger>(w) * row_stride;
        const NSUInteger outOff = oPout + static_cast<NSUInteger>(w) * row_stride;
        run_sardp(dt, bPin, inOff, Wq, bq, Wk, bk, Wv, bv, Wo, bo,
                  rel_pos_h, rel_pos_w, num_heads, ww, D, dh, window, window,
                  scale, bPout, outOff);
    }

    // Scatter per-window results back to the grid (dropping padded tokens).
    run2d(scatter_pso, Du, nrowsu, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bPout offset:oPout atIndex:0];
        [enc setBuffer:buffer_for(O) offset:buffer_offset_for(O) atIndex:1];
        win_args(enc);
    });
}

// ── Transformer-XL relative-position bias ──────────────────────────────────
//
// Bias[h*T + q, k] = sum_d Qv[q, h*head_dim + d] * Pk[(T-1-q) + k, h*head_dim + d]
//
// Ports src/cuda/self_attention_bias.cu's rel_pos_bias_xl_kernel: one
// threadgroup per (head, query) row, the query slice staged in threadgroup
// memory (CUDA's shared-memory `qs`), the rel_shift folded into the pk column
// offset. FP32 only, matching CUDA.
void rel_pos_bias_xl_forward(const Tensor& Qv, const Tensor& Pk,
                             int num_heads, int head_dim, Tensor& Bias) {
    if (Qv.dtype != Dtype::FP32 || Pk.dtype != Dtype::FP32)
        throw std::runtime_error("rel_pos_bias_xl_forward: Qv and Pk must be FP32");
    const int T = Qv.rows;
    const int D = Qv.cols;
    if (num_heads <= 0 || head_dim <= 0 || num_heads * head_dim != D)
        throw std::runtime_error("rel_pos_bias_xl_forward: num_heads*head_dim "
                                 "must equal Qv.cols");
    if (Pk.cols != D || Pk.rows != 2 * T - 1)
        throw std::runtime_error("rel_pos_bias_xl_forward: Pk must be "
                                 "(2*Qv.rows - 1, Qv.cols)");
    if (head_dim > kRelPosXlMaxHeadDim)
        throw std::runtime_error("rel_pos_bias_xl_forward: head_dim above 512 "
                                 "is not implemented on Metal");
    if (Bias.rows != num_heads * T || Bias.cols != T ||
        Bias.dtype != Dtype::FP32) {
        Bias = Tensor::empty_on(Qv.device, num_heads * T, T, Dtype::FP32);
    }
    if (T <= 0) return;

    id<MTLBuffer> bQv = buffer_for(Qv); NSUInteger oQv = buffer_offset_for(Qv);
    id<MTLBuffer> bPk = buffer_for(Pk); NSUInteger oPk = buffer_offset_for(Pk);
    id<MTLBuffer> bB  = buffer_for(Bias);
    NSUInteger oB = buffer_offset_for(Bias);

    const uint32_t Tu  = static_cast<uint32_t>(T);
    const uint32_t Du  = static_cast<uint32_t>(D);
    const uint32_t hdu = static_cast<uint32_t>(head_dim);

    run_rows(pso_rel_pos_bias_xl(),
             static_cast<NSUInteger>(num_heads) * static_cast<NSUInteger>(T),
             ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bQv offset:oQv atIndex:0];
        [enc setBuffer:bPk offset:oPk atIndex:1];
        [enc setBuffer:bB  offset:oB  atIndex:2];
        [enc setBytes:&Tu  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&Du  length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&hdu length:sizeof(uint32_t) atIndex:5];
    });
}

} // namespace brotensor::detail::metal
