// Operation codes shared by the GLSL kernels and the C++ that launches them.
// Plain #defines so the same file is valid GLSL and C++; a kernel receives
// its code as specialisation constant 0, so every (kernel, code) pair is its
// own pipeline with the switch folded away.
#ifndef BT_VK_OP_CODES_H
#define BT_VK_OP_CODES_H

// unary.comp: y = f(x [, a, b])
#define UOP_COPY        0
#define UOP_RELU        1
#define UOP_TANH        2
#define UOP_SIGMOID     3
#define UOP_SILU        4
#define UOP_GELU_TANH   5
#define UOP_GELU_EXACT  6
#define UOP_QUICK_GELU  7
#define UOP_EXP         8
#define UOP_LOG         9
#define UOP_SIN         10
#define UOP_COS         11
#define UOP_RSQRT       12
#define UOP_ROUND       13
#define UOP_ELU         14   // a = alpha
#define UOP_LEAKY_RELU  15   // a = negative slope
#define UOP_ADD_SCALAR  16   // a = s
#define UOP_SCALE       17   // a = s
#define UOP_CLAMP       18   // a = lo, b = hi

// binary.comp: y = f(y, x [, a, b])
#define BOP_ADD         0
#define BOP_MUL         1
#define BOP_DIV         2
#define BOP_AXPBY       3    // a * y + b * x

// act_bwd.comp: dX = g(p, dY), p is x or y as the op's contract says
#define GOP_RELU        0    // p = x
#define GOP_TANH        1    // p = y
#define GOP_SIGMOID     2    // p = y
#define GOP_SILU        3    // p = x
#define GOP_GELU_TANH   4    // p = x
#define GOP_GELU_EXACT  5    // p = x
#define GOP_QUICK_GELU  6    // p = x
#define GOP_EXP         7    // p = x
#define GOP_LOG         8    // p = x, dX = dY / x
#define GOP_SIN         9    // p = x
#define GOP_COS         10   // p = x
#define GOP_RSQRT       11   // p = y
#define GOP_ELU         12   // p = x, a = alpha
#define GOP_LEAKY_RELU  13   // p = x, a = slope
#define GOP_PASS        14   // dX = dY (round's straight-through estimator)

// bias.comp
#define BIAS_ROW        0    // y[i] += bias[i % D]
#define BIAS_CHANNEL    1    // y[i] += bias[(i / L) % C]

// GEMM epilogue activation (= brotensor::LinearActivation) and output mode
// (= brotensor::LinearEpilogue without the fast-accum flag).
#define LACT_NONE       0
#define LACT_RELU       1
#define LACT_GELU_TANH  2
#define LACT_GELU_EXACT 3
#define LACT_SILU       4
#define LACT_QUICK_GELU 5

#define EPI_STORE       0
#define EPI_ACCUM       1
#define EPI_GEGLU       2
#define EPI_SWIGLU      3

// glu.comp
#define GLU_SWIGLU          0
#define GLU_GEGLU_TANH      1
#define GLU_GEGLU_EXACT     2
#define GLU_GEGLU_PAIRS     3
#define GLU_SWIGLU_BWD      4
#define GLU_GEGLU_TANH_BWD  5
#define GLU_GEGLU_EXACT_BWD 6

// norm.comp
#define NORM_LN         0
#define NORM_RMS        1
#define NORM_L2         2
#define NORM_LN_BWD     3
#define NORM_RMS_BWD    4
#define NORM_L2_BWD     5
#define NORM_SOFTMAX    6
#define NORM_SM_BWD     7
#define NORM_COL_LN     8
#define NORM_COL_RMS    9
#define NORM_COL_SUM    10   // g += column sums of x

// rowvec.comp: y = f(x, v[col] [, w[col]])
#define ROWVEC_MODULATE 0    // x * (1 + v) + w
#define ROWVEC_MUL      1    // x * v

// rope.comp
#define ROPE_THETA      0    // angles from theta_base, position = row + offset
#define ROPE_TABLE      1    // cos / sin tables (L, half)
#define ROPE_PERHEAD    2    // tables (L * heads, half)
#define ROPE_PACKED     3    // in place over Q and K of (L, 3 D), tables indexed by pos[row]
#define ROPE_MROPE      4    // three tables / position streams over sub-ranges of the pairs

// fa_cm.comp / fa_rows.comp: how a query row's key interval is found
// (fa_common.glsl fa_interval)
#define FA_MODE_ROWS    0    // q_offset + causal / window over [0, lk)
#define FA_MODE_VARLEN  1    // cu_seqlens_q / cu_seqlens_k (+ causal within the sequence)
#define FA_MODE_PACKED  2    // per-row seq_bounds (+ |q - k| <= window / 2)

// attn_aux.comp
#define AUX_ROW_GATE    0    // y[r, c] = mask[r] >= 0.5 ? x[r, c] : 0 (FP32 x, y in DT)
#define AUX_HEAD_MEAN   1    // y[i] = mean_h x[h * n + i]

// resample.comp (spec constant 0)
#define RS_INTERP           0    // nearest / bilinear / bicubic, half-pixel or corner-aligned
#define RS_DOWN_AVG2        1    // 2x2 average
#define RS_UP_NEAREST2_BWD  2    // dX = sum of the 2x2 dY block
#define RS_DOWN_AVG2_BWD    3    // dX = dY / 4
#define RS_ADAPTIVE_AVG     4    // adaptive average pool
#define RS_MAXPOOL          5    // max pool + INT32 flat-spatial argmax
#define RS_CONVEX           6    // RAFT convex upsample (softmax over 9 neighbours)

// remap.comp: pure gathers (spec constant 0), bit-exact for any element size
#define RM_PAD              0    // zero / reflect / replicate padding (and slice2d_backward)
#define RM_SLICE            1    // crop
#define RM_UP2              2    // nearest 2x upsample
#define RM_UNFOLD           3    // spatial-preserving im2col
#define RM_MERGE2           4    // 2x2 pixel-unshuffle into channels
#define RM_PSHUF            5    // DC-AE repeat_interleave + 2x pixel shuffle
#define RM_PATCH            6    // DiT unpatchify
#define RM_WIN_PART         7    // SAM window partition
#define RM_WIN_REV          8    // SAM window reverse
#define RM_GATHER_ROWS      9    // Y[m, :] = X[clamp(Idx[m]), :]
#define RM_SCATTER_ROWS     10   // X[Idx[m], :] = Y[m, :] (out-of-range rows skipped)

// gnorm.comp (spec constant 0)
#define GN_STATS            0    // partial (count, mean, M2) per (sample, group, split)
#define GN_APPLY            1    // combine the partials, normalise, affine (+ SiLU)
#define GN_BN_INFER         2    // BatchNorm with running statistics
#define GN_L2_NCHW          3    // per-pixel L2 normalise over channels
#define GN_BIAS_GRAD        4    // dB[c] += sum over (n, hw) of dY (conv bias gradient)

// bnorm.comp (spec constant 0)
#define BN_TRAIN_FWD        0    // batch statistics, Y, saved mean / rstd, running-stat update
#define BN_TRAIN_BWD        1    // dX, dGamma += , dBeta +=

// sampler.comp (spec constant 0)
#define SMP_DDIM            0
#define SMP_EULER           1
#define SMP_DPMPP_2M        2
#define SMP_TIMESTEP_EMB    3
#define SMP_IMAGE_NORM      4    // (x - mean[c]) / std[c]
#define SMP_U8_NHWC         5    // u8 NHWC -> NCHW, x * scale + bias

// philox.comp (spec constant 0)
#define RNG_NORMAL          0
#define RNG_UNIFORM         1
#define RNG_BERNOULLI       2
#define RNG_TRUNCATED       3

// Quantised weight formats (quant_decode.glsl, gemv_q.comp, gemm_cm.comp's QB,
// conv_cm.comp's QA, dequant.comp); 0 means not quantised.
#define QF_INT8             1    // int8, per-row FP32 scale
#define QF_Q8_0             2    // GGUF Q8_0
#define QF_Q4K              3    // GGUF Q4_K
#define QF_Q6K              4    // GGUF Q6_K

// audio.comp (spec constant 0)
#define AU_SNAKE            0
#define AU_SNAKE_BWD        1
#define AU_PAD              2
#define AU_PAD_BWD          3
#define AU_RESAMPLE         4
#define AU_RESAMPLE_BWD     5
#define AU_CAUSAL           6
#define AU_CONVT_BWD_W      7
#define AU_FSQ              8
#define AU_VQ               9
#define AU_COL2IM           10

// dft.comp (spec constant 0) and SP_BASIS's modes
#define SP_BASIS            0
#define SP_OLA              1
#define SP_STFT_ADJ         2
#define SP_ISTFT_ADJ        3
#define SP_CX_MUL           4
#define SP_CX_MUL_BWD       5
#define SP_CX_ABS           6
#define SP_CX_ABS_BWD       7
#define SP_CX_ANGLE         8
#define SP_CX_POLAR         9
#define BASIS_R2C           0
#define BASIS_C2R           1
#define BASIS_C2C           2

// select.comp (spec constant 0)
#define SEL_SAMPLE          0
#define SEL_COUNTER         1
#define SEL_MD              2
#define SEL_COMMIT          3

// misc.comp (spec constant 0): the chunk-6 small ops (ops_misc.cpp,
// ops_vision.cpp)
#define MI_MEAN_POOL        0    // y[j] = mean over valid rows of x[:, j]
#define MI_MEAN_POOL_BWD    1    // dx[k, j] = valid(k) ? dy[j] / n_valid : 0
#define MI_SLOT_MASK        2    // mask[k] = x[off + k stride] > 0.5
#define MI_CAUSAL_ROW       3    // mask[k] = k <= q
#define MI_THRESHOLD_U8     4    // y[i] = x[i] > t (INT8 0 / 1)
#define MI_COUNT_ABOVE      5    // counts[r] = (#x > t_lo, #x > t_hi), one workgroup per row
#define MI_XAVIER           6    // splitmix64 uniform in [-limit, limit]
#define MI_SGD              7
#define MI_ADAM             8
#define MI_MSE_SAMPLE       9    // d = p - t, dp = d, loss = 0.5 d^2
#define MI_MSE_BWD          10   // dp = (2 / n) (p - t)
#define MI_MSE_SUM          11   // one workgroup: out[0] = sum (p - t)^2
#define MI_MOMENTS          12   // attention_token_moments, one workgroup per key
#define MI_BIAS_ACT         13   // StyleGAN3 bias_act
#define MI_BIAS_ACT_BWD     14
#define MI_BIAS_ACT_DB      15   // dB[c] += sum over (n, k) of the gradient, one workgroup per channel
#define MI_UPFIRDN          16   // upfirdn2d
#define MI_MODW             17   // modulated_conv2d's per-sample weights, one workgroup per (n, o)
#define MI_COL2IM2D         18   // conv_transpose2d's overlap-add
#define MI_DEFORM_COL       19   // deform_conv2d's bilinear im2col (no offsets: plain im2col)
#define MI_MODW_BWD         20   // modulated_conv2d_backward: dw'' -> dw' through the demodulation
#define MI_MODW_DS          21   // ds[n, i] = sum over (o, t) of dw'[n, o, i, t] W[o, i, t]
#define MI_MODW_DW          22   // dW[o, j] += sum over n of dw'[n, o, j] s[n, j / kk]

#endif
