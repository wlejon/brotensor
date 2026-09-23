// native_tensor_extra_conv.cpp — the image / convolution half of the
// never-bound ops: the StyleGAN3-R set (bias_act, upfirdn2d, modulated_conv2d,
// filtered_lrelu, forward + backward each), torchvision-style deformable
// conv2d, the DC-AE 2x pixel-shuffle upsample and the DiT unpatchify.
//
// Same mechanism as native_tensor_extra.cpp: every NCHW operand is checked
// against N*C*H*W (and every derived output extent against the int the ops
// size tensors with) before the op runs. Accumulated gradients (dB, dW) are
// read by the op, so they are checked like inputs.

#include "native_tensor_extra_decl.h"
#include "native_tensor_extra_util.h"
#include "api_internal.h"
#include "native_register.h"

#include <string>

using namespace brotensor::api;
using namespace brotensor::api::extra;

namespace {

// upfirdn2d's output extent along one axis:
//   ((in*up + pad0 + pad1) - f) / down + 1, which needs the padded input to
// cover the filter.
bool firExtent(const char* lb, const char* axis, int64_t in, int64_t up, int64_t down, int64_t pad0, int64_t pad1,
               int64_t f, int64_t& out) {
    const int64_t padded = in * up + pad0 + pad1;
    if (padded < f) {
        setError(std::string(lb) + ": the padded " + axis + " extent " + std::to_string(padded) +
                 " is smaller than the filter (" + std::to_string(f) + ")");
        return false;
    }
    out = (padded - f) / down + 1;
    return needExtent(lb, axis, out);
}

// The upfirdn2d argument block shared by the forward and backward: N, C, H, W
// positive, up/down >= 1, f (fH, fW) of X's dtype. Answers the output extents.
bool firArgs(const char* lb, brotensor::Tensor* X, const char* xn, brotensor::Tensor* f,
             int32_t N, int32_t C, int32_t H, int32_t W, int32_t fH, int32_t fW,
             int32_t upX, int32_t upY, int32_t downX, int32_t downY,
             int32_t padX0, int32_t padX1, int32_t padY0, int32_t padY1, int64_t& Ho, int64_t& Wo) {
    if (!needFloat(lb, xn, X) || !needSameDtype(lb, "f", f, X)) return false;
    if (N < 1 || C < 1 || H < 1 || W < 1 || fH < 1 || fW < 1) {
        setError(std::string(lb) + ": N, C, H, W, fH and fW must be >= 1");
        return false;
    }
    if (upX < 1 || upY < 1 || downX < 1 || downY < 1) {
        setError(std::string(lb) + ": up / down factors must be >= 1");
        return false;
    }
    if (f->rows != fH || f->cols != fW) {
        setError(std::string(lb) + ": f must be (fH, fW) = (" + std::to_string(fH) + ", " + std::to_string(fW) + ")");
        return false;
    }
    return firExtent(lb, "H_out", H, upY, downY, padY0, padY1, fH, Ho) &&
           firExtent(lb, "W_out", W, upX, downX, padX0, padX1, fW, Wo) &&
           needExtent(lb, "C*H_out*W_out", prod({C, Ho, Wo}));
}

// modulated_conv2d's shared argument block: X (N, C_in*H*W), W (C_out,
// C_in*kH*kW), s (N, C_in) of one float dtype; answers H_out / W_out.
bool modConvArgs(const char* lb, brotensor::Tensor* X, brotensor::Tensor* Wt, brotensor::Tensor* s,
                 int32_t N, int32_t Cin, int32_t H, int32_t Wd, int32_t Cout, int32_t kH, int32_t kW,
                 int32_t padH, int32_t padW, int64_t& Ho, int64_t& Wo) {
    if (!needFloat(lb, "X", X) || !needSameDtype(lb, "W", Wt, X) || !needSameDtype(lb, "s", s, X)) return false;
    if (N < 1 || Cin < 1 || H < 1 || Wd < 1 || Cout < 1 || kH < 1 || kW < 1 || padH < 0 || padW < 0) {
        setError(std::string(lb) + ": dims and kernel must be >= 1, padding >= 0");
        return false;
    }
    Ho = static_cast<int64_t>(H) + 2 * static_cast<int64_t>(padH) - (kH - 1);
    Wo = static_cast<int64_t>(Wd) + 2 * static_cast<int64_t>(padW) - (kW - 1);
    return needExtent(lb, "H_out", Ho) && needExtent(lb, "W_out", Wo) &&
           needExtent(lb, "C_out*H_out*W_out", prod({Cout, Ho, Wo})) &&
           needElems(lb, "X", X, prod({N, Cin, H, Wd})) &&
           needElems(lb, "W", Wt, prod({Cout, Cin, kH, kW})) &&
           needElems(lb, "s", s, prod({N, Cin}));
}

// filtered_lrelu's shared argument block. Answers the post-upsample extents
// (Hu, Wu — the up_buf grid) and the output extents (Ho, Wo).
bool flreluArgs(const char* lb, brotensor::Tensor* X, brotensor::Tensor* fu, brotensor::Tensor* fd,
                brotensor::Tensor* b, int32_t N, int32_t C, int32_t H, int32_t W, int32_t up, int32_t down,
                int32_t padX0, int32_t padX1, int32_t padY0, int32_t padY1,
                int64_t& Hu, int64_t& Wu, int64_t& Ho, int64_t& Wo) {
    if (!needFloat(lb, "X", X) || !needSameDtype(lb, "fu", fu, X) || !needSameDtype(lb, "fd", fd, X)) return false;
    if (N < 1 || C < 1 || H < 1 || W < 1 || up < 1 || down < 1) {
        setError(std::string(lb) + ": N, C, H, W, up and down must be >= 1");
        return false;
    }
    if (fu->empty() || fd->empty()) { setError(std::string(lb) + ": fu and fd must be non-empty filters"); return false; }
    if (!needElems(lb, "X", X, prod({N, C, H, W})) || !optElems(lb, "b", b, C, X)) return false;
    if (!firExtent(lb, "upsampled H", H, up, 1, padY0, padY1, fu->rows, Hu) ||
        !firExtent(lb, "upsampled W", W, up, 1, padX0, padX1, fu->cols, Wu) ||
        !needExtent(lb, "C*H_up*W_up", prod({C, Hu, Wu})) ||
        !firExtent(lb, "H_out", Hu, 1, down, 0, 0, fd->rows, Ho) ||
        !firExtent(lb, "W_out", Wu, 1, down, 0, 0, fd->cols, Wo)) return false;
    return true;
}

} // namespace

extern "C" {

// ---- bias_act ----------------------------------------------------------------

void bro_tensor_biasActForward(void* X, uint64_t b_bits, int32_t N, int32_t C, int32_t HW, int32_t act,
                               double alpha, double gain, double clamp, void* Y) {
    constexpr const char* lb = "biasActForward";
    if (!need(lb, {X, Y})) return;
    auto* x = toTensor(X);
    auto* b = tensorFromValue(b_bits);
    if (!needFloat(lb, "X", x) || !needRange(lb, "act", act, 0, 1)) return;
    if (N < 1 || C < 1 || HW < 1) { setError(std::string(lb) + ": N, C and HW must be >= 1"); return; }
    if (!needElems(lb, "X", x, prod({N, C, HW})) || !optElems(lb, "b", b, C, x)) return;
    BROTENSOR_API_TRY
        brotensor::bias_act_forward(*x, b, N, C, HW, act, static_cast<float>(alpha), static_cast<float>(gain),
                                    static_cast<float>(clamp), *toTensor(Y));
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_biasActBackward(void* dY, void* X, uint64_t b_bits, int32_t N, int32_t C, int32_t HW, int32_t act,
                                double alpha, double gain, double clamp, void* dX, uint64_t dB_bits) {
    constexpr const char* lb = "biasActBackward";
    if (!need(lb, {dY, X, dX})) return;
    auto* dy = toTensor(dY);
    auto* x = toTensor(X);
    auto* b = tensorFromValue(b_bits);
    auto* dB = tensorFromValue(dB_bits);
    if (!needFloat(lb, "X", x) || !needSameDtype(lb, "dY", dy, x) || !needRange(lb, "act", act, 0, 1)) return;
    if (N < 1 || C < 1 || HW < 1) { setError(std::string(lb) + ": N, C and HW must be >= 1"); return; }
    const int64_t n = prod({N, C, HW});
    if (!needElems(lb, "X", x, n) || !needElems(lb, "dY", dy, n) || !optElems(lb, "b", b, C, x) ||
        !optElems(lb, "dB", dB, C, x)) return;
    BROTENSOR_API_TRY
        brotensor::bias_act_backward(*dy, *x, b, N, C, HW, act, static_cast<float>(alpha), static_cast<float>(gain),
                                     static_cast<float>(clamp), *toTensor(dX), dB);
    BROTENSOR_API_CATCH(lb)
}

// ---- upfirdn2d ---------------------------------------------------------------

void bro_tensor_upfirdn2dForward(void* X, void* f, int32_t N, int32_t C, int32_t H, int32_t W, int32_t fH,
                                 int32_t fW, int32_t upX, int32_t upY, int32_t downX, int32_t downY,
                                 int32_t padX0, int32_t padX1, int32_t padY0, int32_t padY1, bool flipFilter,
                                 double gain, void* Y) {
    constexpr const char* lb = "upfirdn2dForward";
    if (!need(lb, {X, f, Y})) return;
    int64_t Ho = 0, Wo = 0;
    if (!firArgs(lb, toTensor(X), "X", toTensor(f), N, C, H, W, fH, fW, upX, upY, downX, downY,
                 padX0, padX1, padY0, padY1, Ho, Wo) ||
        !needElems(lb, "X", toTensor(X), prod({N, C, H, W}))) return;
    BROTENSOR_API_TRY
        brotensor::upfirdn2d_forward(*toTensor(X), *toTensor(f), N, C, H, W, fH, fW, upX, upY, downX, downY,
                                     padX0, padX1, padY0, padY1, flipFilter, static_cast<float>(gain), *toTensor(Y));
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_upfirdn2dBackward(void* dY, void* f, int32_t N, int32_t C, int32_t H, int32_t W, int32_t fH,
                                  int32_t fW, int32_t upX, int32_t upY, int32_t downX, int32_t downY,
                                  int32_t padX0, int32_t padX1, int32_t padY0, int32_t padY1, bool flipFilter,
                                  double gain, void* dX) {
    constexpr const char* lb = "upfirdn2dBackward";
    if (!need(lb, {dY, f, dX})) return;
    int64_t Ho = 0, Wo = 0;
    if (!firArgs(lb, toTensor(dY), "dY", toTensor(f), N, C, H, W, fH, fW, upX, upY, downX, downY,
                 padX0, padX1, padY0, padY1, Ho, Wo) ||
        !needExtent(lb, "C*H*W", prod({C, H, W})) ||
        !needElems(lb, "dY", toTensor(dY), prod({N, C, Ho, Wo}))) return;
    BROTENSOR_API_TRY
        brotensor::upfirdn2d_backward(*toTensor(dY), *toTensor(f), N, C, H, W, fH, fW, upX, upY, downX, downY,
                                      padX0, padX1, padY0, padY1, flipFilter, static_cast<float>(gain),
                                      *toTensor(dX));
    BROTENSOR_API_CATCH(lb)
}

// ---- modulated_conv2d --------------------------------------------------------

void bro_tensor_modulatedConv2dForward(void* X, void* W, void* s, int32_t N, int32_t Cin, int32_t H, int32_t Wd,
                                       int32_t Cout, int32_t kH, int32_t kW, int32_t padH, int32_t padW,
                                       bool demodulate, double eps, void* dcoef, void* Y) {
    constexpr const char* lb = "modulatedConv2dForward";
    if (!need(lb, {X, W, s, dcoef, Y})) return;
    int64_t Ho = 0, Wo = 0;
    if (!modConvArgs(lb, toTensor(X), toTensor(W), toTensor(s), N, Cin, H, Wd, Cout, kH, kW, padH, padW, Ho, Wo))
        return;
    BROTENSOR_API_TRY
        brotensor::modulated_conv2d_forward(*toTensor(X), *toTensor(W), *toTensor(s), N, Cin, H, Wd, Cout, kH, kW,
                                            padH, padW, demodulate, static_cast<float>(eps), *toTensor(dcoef),
                                            *toTensor(Y));
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_modulatedConv2dBackward(void* X, void* W, void* s, void* dcoef, void* dY, int32_t N, int32_t Cin,
                                        int32_t H, int32_t Wd, int32_t Cout, int32_t kH, int32_t kW, int32_t padH,
                                        int32_t padW, bool demodulate, double eps, void* dX, uint64_t dW_bits,
                                        void* ds) {
    constexpr const char* lb = "modulatedConv2dBackward";
    if (!need(lb, {X, W, s, dcoef, dY, dX, ds})) return;
    int64_t Ho = 0, Wo = 0;
    auto* x = toTensor(X);
    if (!modConvArgs(lb, x, toTensor(W), toTensor(s), N, Cin, H, Wd, Cout, kH, kW, padH, padW, Ho, Wo)) return;
    if (!needDtype(lb, "dcoef", toTensor(dcoef), brotensor::Dtype::FP32) ||
        !needElems(lb, "dcoef", toTensor(dcoef), prod({N, Cout})) ||
        !needSameDtype(lb, "dY", toTensor(dY), x) || !needElems(lb, "dY", toTensor(dY), prod({N, Cout, Ho, Wo})))
        return;
    // dW accumulates, so a committed one is read: check it. null or an empty
    // tensor skips the weight gradient.
    auto* dW = tensorFromValue(dW_bits);
    if (dW && !dW->empty() && (!needSameDtype(lb, "dW", dW, x) || !needElems(lb, "dW", dW, prod({Cout, Cin, kH, kW}))))
        return;
    BROTENSOR_API_TRY
        brotensor::Tensor skip;
        brotensor::Tensor& dWref = (dW && !dW->empty()) ? *dW : skip;
        brotensor::modulated_conv2d_backward(*x, *toTensor(W), *toTensor(s), *toTensor(dcoef), *toTensor(dY), N, Cin,
                                             H, Wd, Cout, kH, kW, padH, padW, demodulate, static_cast<float>(eps),
                                             *toTensor(dX), dWref, *toTensor(ds));
    BROTENSOR_API_CATCH(lb)
}

// ---- filtered_lrelu ----------------------------------------------------------

void bro_tensor_filteredLreluForward(void* X, void* fu, void* fd, uint64_t b_bits, int32_t N, int32_t C, int32_t H,
                                     int32_t W, int32_t up, int32_t down, int32_t padX0, int32_t padX1,
                                     int32_t padY0, int32_t padY1, double gain, double slope, double clamp,
                                     void* upBuf, void* actBuf, void* Y) {
    constexpr const char* lb = "filteredLreluForward";
    if (!need(lb, {X, fu, fd, upBuf, actBuf, Y})) return;
    int64_t Hu = 0, Wu = 0, Ho = 0, Wo = 0;
    auto* b = tensorFromValue(b_bits);
    if (!flreluArgs(lb, toTensor(X), toTensor(fu), toTensor(fd), b, N, C, H, W, up, down, padX0, padX1, padY0, padY1,
                    Hu, Wu, Ho, Wo)) return;
    BROTENSOR_API_TRY
        brotensor::filtered_lrelu_forward(*toTensor(X), *toTensor(fu), *toTensor(fd), b, N, C, H, W, up, down,
                                          padX0, padX1, padY0, padY1, static_cast<float>(gain),
                                          static_cast<float>(slope), static_cast<float>(clamp), *toTensor(upBuf),
                                          *toTensor(actBuf), *toTensor(Y));
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_filteredLreluBackward(void* dY, void* X, void* fu, void* fd, uint64_t b_bits, int32_t N, int32_t C,
                                      int32_t H, int32_t W, int32_t up, int32_t down, int32_t padX0, int32_t padX1,
                                      int32_t padY0, int32_t padY1, double gain, double slope, double clamp,
                                      uint64_t upBuf_bits, void* dX, uint64_t dB_bits) {
    constexpr const char* lb = "filteredLreluBackward";
    if (!need(lb, {dY, X, fu, fd, dX})) return;
    int64_t Hu = 0, Wu = 0, Ho = 0, Wo = 0;
    auto* x = toTensor(X);
    auto* b = tensorFromValue(b_bits);
    if (!flreluArgs(lb, x, toTensor(fu), toTensor(fd), b, N, C, H, W, up, down, padX0, padX1, padY0, padY1,
                    Hu, Wu, Ho, Wo)) return;
    if (!needSameDtype(lb, "dY", toTensor(dY), x) || !needElems(lb, "dY", toTensor(dY), prod({N, C, Ho, Wo}))) return;
    // up_buf is optional (an empty one is recomputed from X); a committed one
    // is read over the whole post-upsample grid.
    auto* ub = tensorFromValue(upBuf_bits);
    if (ub && !ub->empty() && (!needSameDtype(lb, "upBuf", ub, x) || !needElems(lb, "upBuf", ub, prod({N, C, Hu, Wu}))))
        return;
    auto* dB = tensorFromValue(dB_bits);
    if (!optElems(lb, "dB", dB, C, x)) return;
    BROTENSOR_API_TRY
        brotensor::Tensor none;
        const brotensor::Tensor& upRef = (ub && !ub->empty()) ? *ub : none;
        brotensor::filtered_lrelu_backward(*toTensor(dY), *x, *toTensor(fu), *toTensor(fd), b, N, C, H, W, up, down,
                                           padX0, padX1, padY0, padY1, static_cast<float>(gain),
                                           static_cast<float>(slope), static_cast<float>(clamp), upRef,
                                           *toTensor(dX), dB);
    BROTENSOR_API_CATCH(lb)
}

// ---- deformable conv2d -------------------------------------------------------

void bro_tensor_deformConv2dForward(void* X, void* offset, uint64_t mask_bits, void* Wt, uint64_t bias_bits,
                                    int32_t N, int32_t Cin, int32_t H, int32_t W, int32_t Cout, int32_t kH,
                                    int32_t kW, int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH,
                                    int32_t dW, int32_t groups, int32_t deformGroups, void* Y) {
    constexpr const char* lb = "deformConv2dForward";
    if (!need(lb, {X, offset, Wt, Y})) return;
    auto* x = toTensor(X);
    auto* off = toTensor(offset);
    auto* w = toTensor(Wt);
    auto* mask = tensorFromValue(mask_bits);
    auto* bias = tensorFromValue(bias_bits);
    if (!needFloat(lb, "X", x) || !needSameDtype(lb, "offset", off, x) || !needSameDtype(lb, "Wt", w, x)) return;
    if (N < 1 || Cin < 1 || H < 1 || W < 1 || Cout < 1 || kH < 1 || kW < 1 || sH < 1 || sW < 1 || pH < 0 ||
        pW < 0 || dH < 1 || dW < 1) {
        setError(std::string(lb) + ": dims, kernel, stride and dilation must be >= 1, padding >= 0");
        return;
    }
    if (groups < 1 || Cin % groups != 0 || Cout % groups != 0) {
        setError(std::string(lb) + ": groups must divide C_in and C_out");
        return;
    }
    if (deformGroups < 1 || Cin % deformGroups != 0) {
        setError(std::string(lb) + ": deformGroups must divide C_in");
        return;
    }
    const int64_t Ho = (static_cast<int64_t>(H) + 2 * pH - static_cast<int64_t>(dH) * (kH - 1) - 1) / sH + 1;
    const int64_t Wo = (static_cast<int64_t>(W) + 2 * pW - static_cast<int64_t>(dW) * (kW - 1) - 1) / sW + 1;
    if (!needExtent(lb, "H_out", Ho) || !needExtent(lb, "W_out", Wo) ||
        !needExtent(lb, "C_out*H_out*W_out", prod({Cout, Ho, Wo}))) return;
    const int64_t taps = prod({deformGroups, kH, kW});
    if (!needElems(lb, "X", x, prod({N, Cin, H, W})) ||
        !needElems(lb, "offset", off, prod({N, 2, taps, Ho, Wo})) ||
        !optElems(lb, "mask", mask, prod({N, taps, Ho, Wo}), x) ||
        !needElems(lb, "Wt", w, prod({Cout, Cin / groups, kH, kW})) ||
        !optElems(lb, "bias", bias, Cout, x)) return;
    BROTENSOR_API_TRY
        brotensor::deform_conv2d_forward(*x, *off, mask, *w, bias, N, Cin, H, W, Cout, kH, kW, sH, sW, pH, pW, dH, dW,
                                         groups, deformGroups, *toTensor(Y));
    BROTENSOR_API_CATCH(lb)
}

// ---- pixel-shuffle upsample + unpatchify -------------------------------------

void bro_tensor_pixelShuffleUpsample2xForward(void* X, int32_t N, int32_t Cin, int32_t H, int32_t W, int32_t Cout,
                                              void* Y) {
    constexpr const char* lb = "pixelShuffleUpsample2xForward";
    if (!need(lb, {X, Y})) return;
    auto* x = toTensor(X);
    if (!needFloat(lb, "X", x)) return;
    if (N < 1 || Cin < 1 || H < 1 || W < 1 || Cout < 1 || (4 * static_cast<int64_t>(Cout)) % Cin != 0) {
        setError(std::string(lb) + ": dims must be >= 1 and C_in must divide 4*C_out");
        return;
    }
    if (!needElems(lb, "X", x, prod({N, Cin, H, W})) || !needExtent(lb, "C_out*2H*2W", prod({Cout, 2 * int64_t(H), 2 * int64_t(W)})))
        return;
    BROTENSOR_API_TRY
        brotensor::pixel_shuffle_upsample_2x_forward(*x, N, Cin, H, W, Cout, *toTensor(Y));
    BROTENSOR_API_CATCH(lb)
}

void bro_tensor_patchUnpackForward(void* tokens, int32_t hp, int32_t wp, int32_t P, int32_t Ctotal, int32_t Ckeep,
                                   bool channelMajor, void* Y) {
    constexpr const char* lb = "patchUnpackForward";
    if (!need(lb, {tokens, Y})) return;
    auto* t = toTensor(tokens);
    if (!needFloat(lb, "tokens", t)) return;
    if (hp < 1 || wp < 1 || P < 1 || Ctotal < 1 || Ckeep < 1 || Ckeep > Ctotal) {
        setError(std::string(lb) + ": hp, wp, P, C_total must be >= 1 and 1 <= C_keep <= C_total");
        return;
    }
    const int64_t rowWidth = prod({P, P, Ctotal});
    if (t->cols != rowWidth || t->rows < static_cast<int64_t>(hp) * wp) {
        setError(std::string(lb) + ": tokens must be (hp*wp, P*P*C_total) = (" + std::to_string(int64_t(hp) * wp) +
                 ", " + std::to_string(rowWidth) + ")");
        return;
    }
    if (!needExtent(lb, "C_keep*(hp*P)*(wp*P)", prod({Ckeep, hp, P, wp, P}))) return;
    BROTENSOR_API_TRY
        brotensor::patch_unpack_forward(*t, hp, wp, P, Ctotal, Ckeep, channelMajor, *toTensor(Y));
    BROTENSOR_API_CATCH(lb)
}

} // extern "C"

namespace brotensor::api {

bool registerTensorNatives_extra_conv(std::string* error) {
    using namespace brotensor::api::reg;
    const char* T = kTensorCls;
    return
        fn("__bro_native.tensor.biasActForward", p(&bro_tensor_biasActForward), "void",
           {T, kDyn, "i32", "i32", "i32", "i32", "f64", "f64", "f64", T}, error) &&
        fn("__bro_native.tensor.biasActBackward", p(&bro_tensor_biasActBackward), "void",
           {T, T, kDyn, "i32", "i32", "i32", "i32", "f64", "f64", "f64", T, kDyn}, error) &&
        fn("__bro_native.tensor.upfirdn2dForward", p(&bro_tensor_upfirdn2dForward), "void",
           {T, T, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            "i32", "i32", "i32", "i32", "bool", "f64", T}, error) &&
        fn("__bro_native.tensor.upfirdn2dBackward", p(&bro_tensor_upfirdn2dBackward), "void",
           {T, T, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            "i32", "i32", "i32", "i32", "bool", "f64", T}, error) &&
        fn("__bro_native.tensor.modulatedConv2dForward", p(&bro_tensor_modulatedConv2dForward), "void",
           {T, T, T, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "bool", "f64", T, T}, error) &&
        fn("__bro_native.tensor.modulatedConv2dBackward", p(&bro_tensor_modulatedConv2dBackward), "void",
           {T, T, T, T, T, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "bool", "f64",
            T, kDyn, T}, error) &&
        fn("__bro_native.tensor.filteredLreluForward", p(&bro_tensor_filteredLreluForward), "void",
           {T, T, T, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            "f64", "f64", "f64", T, T, T}, error) &&
        fn("__bro_native.tensor.filteredLreluBackward", p(&bro_tensor_filteredLreluBackward), "void",
           {T, T, T, T, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            "f64", "f64", "f64", kDyn, T, kDyn}, error) &&
        fn("__bro_native.tensor.deformConv2dForward", p(&bro_tensor_deformConv2dForward), "void",
           {T, T, kDyn, T, kDyn, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            "i32", "i32", "i32", "i32", T}, error) &&
        fn("__bro_native.tensor.pixelShuffleUpsample2xForward", p(&bro_tensor_pixelShuffleUpsample2xForward), "void",
           {T, "i32", "i32", "i32", "i32", "i32", T}, error) &&
        fn("__bro_native.tensor.patchUnpackForward", p(&bro_tensor_patchUnpackForward), "void",
           {T, "i32", "i32", "i32", "i32", "i32", "bool", T}, error);
}

} // namespace brotensor::api
