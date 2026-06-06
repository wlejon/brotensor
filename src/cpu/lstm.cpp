// CPU implementation of the trainable LSTM cell (ops/lstm.h) — training
// forward with per-step caching, and full back-propagation-through-time.
//
// PyTorch nn.LSTM weight layout: W_ih (4H,I), W_hh (4H,H), gate-row blocks in
// order [input, forget, cell, output]; gate g uses tanh, the rest sigmoid.
// CPU-resident, FP32 throughout.

#include <brotensor/tensor.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor::detail::cpu {

namespace {

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void require_cpu_f32(const ::brotensor::Tensor& t, const char* what) {
    if (t.device != ::brotensor::Device::CPU ||
        t.dtype != ::brotensor::Dtype::FP32) {
        throw std::runtime_error(std::string("lstm: ") + what +
                                 " must be CPU FP32");
    }
}

// An optional bias/state tensor: present iff non-null and non-empty.
const float* opt_ptr(const ::brotensor::Tensor* t, const char* what,
                     int rows, int cols) {
    if (!t || t->empty()) return nullptr;
    require_cpu_f32(*t, what);
    if (t->rows != rows || t->cols != cols) {
        throw std::runtime_error(std::string("lstm: ") + what + " has wrong shape");
    }
    return t->host_f32();
}

}  // namespace

void lstm_forward_train(const ::brotensor::Tensor& X, const ::brotensor::Tensor& W_ih,
                        const ::brotensor::Tensor& W_hh,
                        const ::brotensor::Tensor* b_ih, const ::brotensor::Tensor* b_hh,
                        const ::brotensor::Tensor* h0, const ::brotensor::Tensor* c0,
                        int T, int B,
                        ::brotensor::Tensor& Y, ::brotensor::Tensor& gates,
                        ::brotensor::Tensor& C,
                        ::brotensor::Tensor* hT, ::brotensor::Tensor* cT) {
    require_cpu_f32(X, "X");
    require_cpu_f32(W_ih, "W_ih");
    require_cpu_f32(W_hh, "W_hh");
    const int H = W_hh.cols;
    const int I = W_ih.cols;
    const int G = 4 * H;
    if (T <= 0 || B <= 0)
        throw std::runtime_error("lstm_forward_train: T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G)
        throw std::runtime_error("lstm_forward_train: W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I)
        throw std::runtime_error("lstm_forward_train: X must be (T*B, I)");

    const float* bih = opt_ptr(b_ih, "b_ih", G, 1);
    const float* bhh = opt_ptr(b_hh, "b_hh", G, 1);
    const float* h0p = opt_ptr(h0, "h0", B, H);
    const float* c0p = opt_ptr(c0, "c0", B, H);

    Y.resize(T * B, H);
    gates.resize(T * B, G);
    C.resize(T * B, H);
    if (hT) hT->resize(B, H);
    if (cT) cT->resize(B, H);

    const float* xp = X.host_f32();
    const float* wih = W_ih.host_f32();
    const float* whh = W_hh.host_f32();
    float* yp = Y.host_f32_mut();
    float* gp = gates.host_f32_mut();
    float* cp = C.host_f32_mut();

    for (int t = 0; t < T; ++t) {
        for (int b = 0; b < B; ++b) {
            const std::size_t row = static_cast<std::size_t>(t) * B + b;
            const float* x = xp + row * I;
            const float* hprev = (t == 0) ? (h0p ? h0p + static_cast<std::size_t>(b) * H : nullptr)
                                          : yp + (static_cast<std::size_t>(t - 1) * B + b) * H;
            const float* cprev = (t == 0) ? (c0p ? c0p + static_cast<std::size_t>(b) * H : nullptr)
                                          : cp + (static_cast<std::size_t>(t - 1) * B + b) * H;
            float* grow = gp + row * G;   // post-activation [i|f|g|o]
            float* crow = cp + row * H;
            float* hrow = yp + row * H;

            for (int n = 0; n < H; ++n) {
                float zi = bih ? bih[n]         : 0.0f;
                float zf = bih ? bih[H + n]     : 0.0f;
                float zg = bih ? bih[2 * H + n] : 0.0f;
                float zo = bih ? bih[3 * H + n] : 0.0f;
                if (bhh) { zi += bhh[n]; zf += bhh[H + n]; zg += bhh[2 * H + n]; zo += bhh[3 * H + n]; }

                const float* wi = wih + static_cast<std::size_t>(n) * I;
                const float* wf = wih + static_cast<std::size_t>(H + n) * I;
                const float* wg = wih + static_cast<std::size_t>(2 * H + n) * I;
                const float* wo = wih + static_cast<std::size_t>(3 * H + n) * I;
                for (int j = 0; j < I; ++j) {
                    const float xj = x[j];
                    zi += wi[j] * xj; zf += wf[j] * xj; zg += wg[j] * xj; zo += wo[j] * xj;
                }
                if (hprev) {
                    const float* ui = whh + static_cast<std::size_t>(n) * H;
                    const float* uf = whh + static_cast<std::size_t>(H + n) * H;
                    const float* ug = whh + static_cast<std::size_t>(2 * H + n) * H;
                    const float* uo = whh + static_cast<std::size_t>(3 * H + n) * H;
                    for (int m = 0; m < H; ++m) {
                        const float hm = hprev[m];
                        zi += ui[m] * hm; zf += uf[m] * hm; zg += ug[m] * hm; zo += uo[m] * hm;
                    }
                }

                const float ig = sigmoidf(zi);
                const float fg = sigmoidf(zf);
                const float gg = std::tanh(zg);
                const float og = sigmoidf(zo);
                const float cprev_n = cprev ? cprev[n] : 0.0f;
                const float cn = fg * cprev_n + ig * gg;

                grow[n] = ig; grow[H + n] = fg; grow[2 * H + n] = gg; grow[3 * H + n] = og;
                crow[n] = cn;
                hrow[n] = og * std::tanh(cn);
            }
        }
    }

    if (hT) {
        float* p = hT->host_f32_mut();
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < H; ++n)
                p[static_cast<std::size_t>(b) * H + n] =
                    yp[(static_cast<std::size_t>(T - 1) * B + b) * H + n];
    }
    if (cT) {
        float* p = cT->host_f32_mut();
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < H; ++n)
                p[static_cast<std::size_t>(b) * H + n] =
                    cp[(static_cast<std::size_t>(T - 1) * B + b) * H + n];
    }
}

void lstm_backward(const ::brotensor::Tensor& X, const ::brotensor::Tensor& W_ih,
                   const ::brotensor::Tensor& W_hh,
                   const ::brotensor::Tensor* h0, const ::brotensor::Tensor* c0,
                   const ::brotensor::Tensor& Y, const ::brotensor::Tensor& gates,
                   const ::brotensor::Tensor& C,
                   const ::brotensor::Tensor& dY, int T, int B,
                   ::brotensor::Tensor& dX, ::brotensor::Tensor& dW_ih,
                   ::brotensor::Tensor& dW_hh,
                   ::brotensor::Tensor* db_ih, ::brotensor::Tensor* db_hh,
                   ::brotensor::Tensor* dh0, ::brotensor::Tensor* dc0) {
    require_cpu_f32(X, "X");
    require_cpu_f32(W_ih, "W_ih");
    require_cpu_f32(W_hh, "W_hh");
    require_cpu_f32(Y, "Y");
    require_cpu_f32(gates, "gates");
    require_cpu_f32(C, "C");
    require_cpu_f32(dY, "dY");
    const int H = W_hh.cols;
    const int I = W_ih.cols;
    const int G = 4 * H;
    if (T <= 0 || B <= 0)
        throw std::runtime_error("lstm_backward: T and B must be > 0");
    if (W_ih.rows != G || W_hh.rows != G)
        throw std::runtime_error("lstm_backward: W_ih/W_hh must have 4*H rows");
    if (X.rows != T * B || X.cols != I)
        throw std::runtime_error("lstm_backward: X must be (T*B, I)");
    if (dY.rows != T * B || dY.cols != H)
        throw std::runtime_error("lstm_backward: dY must be (T*B, H)");

    // Parameter grads accumulate — require pre-sized buffers, do NOT zero them.
    require_cpu_f32(dW_ih, "dW_ih");
    require_cpu_f32(dW_hh, "dW_hh");
    if (dW_ih.rows != G || dW_ih.cols != I)
        throw std::runtime_error("lstm_backward: dW_ih must be (4H, I), zeroed");
    if (dW_hh.rows != G || dW_hh.cols != H)
        throw std::runtime_error("lstm_backward: dW_hh must be (4H, H), zeroed");
    if (db_ih) { require_cpu_f32(*db_ih, "db_ih"); if (db_ih->rows != G || db_ih->cols != 1) throw std::runtime_error("lstm_backward: db_ih must be (4H, 1), zeroed"); }
    if (db_hh) { require_cpu_f32(*db_hh, "db_hh"); if (db_hh->rows != G || db_hh->cols != 1) throw std::runtime_error("lstm_backward: db_hh must be (4H, 1), zeroed"); }

    const float* h0p = opt_ptr(h0, "h0", B, H);
    const float* c0p = opt_ptr(c0, "c0", B, H);

    dX.resize(T * B, I);
    if (dh0) dh0->resize(B, H);
    if (dc0) dc0->resize(B, H);

    const float* xp = X.host_f32();
    const float* yp = Y.host_f32();
    const float* gp = gates.host_f32();
    const float* cp = C.host_f32();
    const float* dyp = dY.host_f32();
    const float* wih = W_ih.host_f32();
    const float* whh = W_hh.host_f32();
    float* dxp = dX.host_f32_mut();
    float* dwih = dW_ih.host_f32_mut();
    float* dwhh = dW_hh.host_f32_mut();
    float* dbih = db_ih ? db_ih->host_f32_mut() : nullptr;
    float* dbhh = db_hh ? db_hh->host_f32_mut() : nullptr;

    std::fill(dxp, dxp + static_cast<std::size_t>(T) * B * I, 0.0f);

    // Recurrent grad carried backward in time: dL/dh_{t} and dL/dc_{t} arriving
    // from step t+1. After the loop these hold dL/dh0 and dL/dc0.
    std::vector<float> dh_next(static_cast<std::size_t>(B) * H, 0.0f);
    std::vector<float> dc_next(static_cast<std::size_t>(B) * H, 0.0f);
    std::vector<float> dh_new(static_cast<std::size_t>(H));

    for (int t = T - 1; t >= 0; --t) {
        for (int b = 0; b < B; ++b) {
            const std::size_t row = static_cast<std::size_t>(t) * B + b;
            const float* grow = gp + row * G;
            const float* crow = cp + row * H;
            const float* x = xp + row * I;
            const float* dyrow = dyp + row * H;
            const float* hprev = (t == 0) ? (h0p ? h0p + static_cast<std::size_t>(b) * H : nullptr)
                                          : yp + (static_cast<std::size_t>(t - 1) * B + b) * H;
            const float* cprev = (t == 0) ? (c0p ? c0p + static_cast<std::size_t>(b) * H : nullptr)
                                          : cp + (static_cast<std::size_t>(t - 1) * B + b) * H;
            float* dxrow = dxp + row * I;
            float* dh_b = dh_next.data() + static_cast<std::size_t>(b) * H;
            float* dc_b = dc_next.data() + static_cast<std::size_t>(b) * H;

            std::fill(dh_new.begin(), dh_new.end(), 0.0f);

            for (int n = 0; n < H; ++n) {
                const float ig = grow[n], fg = grow[H + n], gg = grow[2 * H + n], og = grow[3 * H + n];
                const float tc = std::tanh(crow[n]);
                const float dh = dyrow[n] + dh_b[n];
                const float do_ = dh * tc;
                const float dc = dh * og * (1.0f - tc * tc) + dc_b[n];
                const float cprev_n = cprev ? cprev[n] : 0.0f;
                const float df = dc * cprev_n;
                const float di = dc * gg;
                const float dg = dc * ig;
                const float dcprev = dc * fg;

                // Gate pre-activation grads (sigmoid'/tanh' on cached outputs).
                const float dzi = di * ig * (1.0f - ig);
                const float dzf = df * fg * (1.0f - fg);
                const float dzg = dg * (1.0f - gg * gg);
                const float dzo = do_ * og * (1.0f - og);

                if (dbih) { dbih[n] += dzi; dbih[H + n] += dzf; dbih[2 * H + n] += dzg; dbih[3 * H + n] += dzo; }
                if (dbhh) { dbhh[n] += dzi; dbhh[H + n] += dzf; dbhh[2 * H + n] += dzg; dbhh[3 * H + n] += dzo; }

                // Input path: dX = W_ih^T·dz ;  dW_ih += dz·x^T
                const float* wi = wih + static_cast<std::size_t>(n) * I;
                const float* wf = wih + static_cast<std::size_t>(H + n) * I;
                const float* wg = wih + static_cast<std::size_t>(2 * H + n) * I;
                const float* wo = wih + static_cast<std::size_t>(3 * H + n) * I;
                float* dwi = dwih + static_cast<std::size_t>(n) * I;
                float* dwf = dwih + static_cast<std::size_t>(H + n) * I;
                float* dwg = dwih + static_cast<std::size_t>(2 * H + n) * I;
                float* dwo = dwih + static_cast<std::size_t>(3 * H + n) * I;
                for (int j = 0; j < I; ++j) {
                    dxrow[j] += wi[j] * dzi + wf[j] * dzf + wg[j] * dzg + wo[j] * dzo;
                    const float xj = x[j];
                    dwi[j] += dzi * xj; dwf[j] += dzf * xj; dwg[j] += dzg * xj; dwo[j] += dzo * xj;
                }

                // Recurrent path: dh_prev = W_hh^T·dz ;  dW_hh += dz·h_prev^T
                const float* ui = whh + static_cast<std::size_t>(n) * H;
                const float* uf = whh + static_cast<std::size_t>(H + n) * H;
                const float* ug = whh + static_cast<std::size_t>(2 * H + n) * H;
                const float* uo = whh + static_cast<std::size_t>(3 * H + n) * H;
                float* dui = dwhh + static_cast<std::size_t>(n) * H;
                float* duf = dwhh + static_cast<std::size_t>(H + n) * H;
                float* dug = dwhh + static_cast<std::size_t>(2 * H + n) * H;
                float* duo = dwhh + static_cast<std::size_t>(3 * H + n) * H;
                for (int m = 0; m < H; ++m) {
                    dh_new[m] += ui[m] * dzi + uf[m] * dzf + ug[m] * dzg + uo[m] * dzo;
                    if (hprev) {
                        const float hm = hprev[m];
                        dui[m] += dzi * hm; duf[m] += dzf * hm; dug[m] += dzg * hm; duo[m] += dzo * hm;
                    }
                }

                // dL/dc_{t-1} for this unit (cell carry is per-unit, no coupling).
                dc_b[n] = dcprev;
            }

            // Swap in the freshly accumulated dL/dh_{t-1} for the earlier step.
            for (int m = 0; m < H; ++m) dh_b[m] = dh_new[m];
        }
    }

    if (dh0) {
        float* p = dh0->host_f32_mut();
        for (std::size_t i = 0; i < dh_next.size(); ++i) p[i] = dh_next[i];
    }
    if (dc0) {
        float* p = dc0->host_f32_mut();
        for (std::size_t i = 0; i < dc_next.size(); ++i) p[i] = dc_next[i];
    }
}

}  // namespace brotensor::detail::cpu
