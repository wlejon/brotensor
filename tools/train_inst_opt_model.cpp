#include "inst_opt_dataset.hpp"

#include <brass/mir/inst_opt_model.hpp>
#include <brotensor/ops/activation.h>
#include <brotensor/ops/elementwise.h>
#include <brotensor/ops/linear.h>
#include <brotensor/ops/optim.h>
#include <brotensor/runtime.h>
#include <brotensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace brotensor {
namespace {

struct TrainConfig {
    std::string out_brim = "D:/projects/brass/data/inst_opt_model_v1.brim";
    std::string out_inc  = "D:/projects/brass/src/mir/inst_opt_weights_builtin.inc";
    std::string device_arg = "auto";
    int epochs = 160;
    int batch_size = 128;
    uint64_t seed = 0xB4A55101ULL;
};

TrainConfig parse_args(int argc, char** argv) {
    TrainConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_val = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
            return argv[++i];
        };
        if (arg == "--out-brim") cfg.out_brim = next_val(arg);
        else if (arg == "--out-inc") cfg.out_inc = next_val(arg);
        else if (arg == "--epochs") cfg.epochs = std::stoi(next_val(arg));
        else if (arg == "--batch") cfg.batch_size = std::stoi(next_val(arg));
        else if (arg == "--seed") cfg.seed = std::stoull(next_val(arg));
        else if (arg == "--device") cfg.device_arg = next_val(arg);
        else throw std::runtime_error("Unknown argument: " + arg);
    }
    return cfg;
}

uint64_t lcg(uint64_t& s) noexcept {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s;
}

struct Param {
    Tensor w, dw, m, v;
    Param() = default;
    Param(int rows, int cols, uint64_t& rng_seed, Device dev, bool is_bias = false) {
        Tensor h = Tensor::zeros_on(Device::CPU, rows, cols);
        if (!is_bias) {
            xavier_init(h, rng_seed);
        }
        w = h.to(dev);
        dw = Tensor::zeros_on(dev, rows, cols);
        m = Tensor::zeros_on(dev, rows, cols);
        v = Tensor::zeros_on(dev, rows, cols);
    }
    void zero_grad() { dw.zero(); }
    void step(float lr, int t) {
        // Clip huge gradients on host/device smoothly via adam
        adam_step(w, dw, m, v, lr, 0.9f, 0.999f, 1e-8f, t);
    }
};

struct TrunkActivations {
    Tensor X_dev, H1_pre, H1, H2_pre, H2, H3_pre, H3, Z_cost, Z_pol;
};

struct InstOptNet {
    Device dev;
    Param l1_w, l1_b;
    Param l2_w, l2_b;
    Param l3_w, l3_b;
    Param c_w,  c_b;
    Param p_w,  p_b;

    InstOptNet(Device d, uint64_t seed) : dev(d) {
        constexpr int D = brass::kInstOptInputDim;
        constexpr int H1 = brass::kInstOptHidden1;
        constexpr int H2 = brass::kInstOptHidden2;
        constexpr int H3 = brass::kInstOptHidden3;
        constexpr int NA = brass::kNumInstOptActions;
        uint64_t rng = seed;
        l1_w = Param(H1, D, rng, d);
        l1_b = Param(H1, 1, rng, d, true);
        l2_w = Param(H2, H1, rng, d);
        l2_b = Param(H2, 1, rng, d, true);
        l3_w = Param(H3, H2, rng, d);
        l3_b = Param(H3, 1, rng, d, true);
        c_w  = Param(1, H3, rng, d);
        c_b  = Param(1, 1, rng, d, true);
        p_w  = Param(NA, H3, rng, d);
        p_b  = Param(NA, 1, rng, d, true);
    }

    TrunkActivations forward(const std::vector<float>& host_x, int B) const {
        TrunkActivations a;
        a.X_dev = Tensor::from_host_on(dev, host_x.data(), B, static_cast<int>(brass::kInstOptInputDim));

        linear_forward_batched(l1_w.w, l1_b.w, a.X_dev, a.H1_pre);
        relu_forward_batched(a.H1_pre, a.H1);

        linear_forward_batched(l2_w.w, l2_b.w, a.H1, a.H2_pre);
        relu_forward_batched(a.H2_pre, a.H2);
        add_inplace_batched(a.H2, a.H1); // residual skip: h2 = relu(W2*h1 + b2) + h1

        linear_forward_batched(l3_w.w, l3_b.w, a.H2, a.H3_pre);
        relu_forward_batched(a.H3_pre, a.H3);

        linear_forward_batched(c_w.w, c_b.w, a.H3, a.Z_cost);
        linear_forward_batched(p_w.w, p_b.w, a.H3, a.Z_pol);
        return a;
    }

    void zero_grad() {
        l1_w.zero_grad(); l1_b.zero_grad();
        l2_w.zero_grad(); l2_b.zero_grad();
        l3_w.zero_grad(); l3_b.zero_grad();
        c_w.zero_grad();  c_b.zero_grad();
        p_w.zero_grad();  p_b.zero_grad();
    }

    void backward(const TrunkActivations& a,
                  const std::vector<float>& host_dz_cost,
                  const std::vector<float>& host_dz_pol,
                  int B) {
        constexpr int H1 = brass::kInstOptHidden1;
        constexpr int H2 = brass::kInstOptHidden2;
        constexpr int H3 = brass::kInstOptHidden3;
        constexpr int NA = brass::kNumInstOptActions;

        Tensor dZ_cost = Tensor::from_host_on(dev, host_dz_cost.data(), B, 1);
        Tensor dZ_pol  = Tensor::from_host_on(dev, host_dz_pol.data(), B, NA);

        Tensor dH3_from_c = Tensor::zeros_on(dev, B, H3);
        Tensor dH3_from_p = Tensor::zeros_on(dev, B, H3);
        linear_backward_batched(c_w.w, a.H3, dZ_cost, dH3_from_c, c_w.dw, c_b.dw);
        linear_backward_batched(p_w.w, a.H3, dZ_pol,  dH3_from_p, p_w.dw, p_b.dw);
        add_inplace_batched(dH3_from_c, dH3_from_p);

        Tensor dH3_pre = Tensor::zeros_on(dev, B, H3);
        relu_backward_batched(a.H3_pre, dH3_from_c, dH3_pre);

        Tensor dH2 = Tensor::zeros_on(dev, B, H2);
        linear_backward_batched(l3_w.w, a.H2, dH3_pre, dH2, l3_w.dw, l3_b.dw);

        // Residual backward splits gradient to both relu(H2_pre) and H1 skip
        Tensor dH2_act = Tensor::zeros_on(dev, B, H2);
        relu_backward_batched(a.H2_pre, dH2, dH2_act);

        Tensor dH1 = Tensor::zeros_on(dev, B, H1);
        linear_backward_batched(l2_w.w, a.H1, dH2_act, dH1, l2_w.dw, l2_b.dw);
        add_inplace_batched(dH1, dH2);

        Tensor dH1_pre = Tensor::zeros_on(dev, B, H1);
        relu_backward_batched(a.H1_pre, dH1, dH1_pre);

        Tensor dX = Tensor::zeros_on(dev, B, static_cast<int>(brass::kInstOptInputDim));
        linear_backward_batched(l1_w.w, a.X_dev, dH1_pre, dX, l1_w.dw, l1_b.dw);
    }

    void adam_update(float lr, int step) {
        l1_w.step(lr, step); l1_b.step(lr, step);
        l2_w.step(lr, step); l2_b.step(lr, step);
        l3_w.step(lr, step); l3_b.step(lr, step);
        c_w.step(lr, step);  c_b.step(lr, step);
        p_w.step(lr, step);  p_b.step(lr, step);
    }

    brass::InstOptModel export_brass_model() const {
        brass::InstOptModel m;
        auto copy_to = [](const Tensor& t, float* dst, size_t n) {
            Tensor h = t.to(Device::CPU);
            std::memcpy(dst, h.host_f32(), sizeof(float) * n);
        };
        copy_to(l1_w.w, m.w1.data(), m.w1.size());
        copy_to(l1_b.w, m.b1.data(), m.b1.size());
        copy_to(l2_w.w, m.w2.data(), m.w2.size());
        copy_to(l2_b.w, m.b2.data(), m.b2.size());
        copy_to(l3_w.w, m.w3.data(), m.w3.size());
        copy_to(l3_b.w, m.b3.data(), m.b3.size());
        copy_to(c_w.w,  m.w_cost.data(), m.w_cost.size());
        Tensor cb_h = c_b.w.to(Device::CPU);
        m.b_cost = cb_h.host_f32()[0];
        copy_to(p_w.w, m.w_pol.data(), m.w_pol.size());
        copy_to(p_b.w, m.b_pol.data(), m.b_pol.size());
        m.loaded = true;
        return m;
    }
};

struct EvalMetrics {
    float cost_mae = 0.0f;
    float rank_acc = 0.0f;
    float static_rank_acc = 0.0f;
    float pol_top1 = 0.0f;
    float pol_top3 = 0.0f;
};

EvalMetrics evaluate_split(const brass::InstOptModel& mdl,
                           const inst_opt_train::InstOptDataset& ds,
                           const std::vector<size_t>& sample_idxs,
                           const std::vector<size_t>& pair_idxs) {
    EvalMetrics em;
    std::vector<float> preds(ds.samples.size(), 0.0f);
    double sum_ae = 0.0;
    size_t top1_hits = 0, top3_hits = 0;

    for (size_t idx : sample_idxs) {
        const auto& s = ds.samples[idx];
        const brass::InstOptPrediction p = mdl.predict(s.features);
        preds[idx] = p.predicted_cost;
        sum_ae += std::fabs(p.predicted_cost - s.measured_cost);

        std::array<uint32_t, brass::kNumInstOptActions> order{};
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return p.action_probs[a] > p.action_probs[b];
        });
        if (order[0] == s.action_label) top1_hits++;
        if (order[0] == s.action_label || order[1] == s.action_label || order[2] == s.action_label) {
            top3_hits++;
        }
    }

    const double n_s = static_cast<double>(std::max<size_t>(1, sample_idxs.size()));
    em.cost_mae = static_cast<float>(sum_ae / n_s);
    em.pol_top1 = static_cast<float>(static_cast<double>(top1_hits) / n_s);
    em.pol_top3 = static_cast<float>(static_cast<double>(top3_hits) / n_s);

    size_t model_ok = 0, static_ok = 0;
    for (size_t pid : pair_idxs) {
        const auto& pr = ds.pairs[pid];
        const float pf = mdl.predict_cost(ds.samples[pr.fast_idx].features);
        const float ps = mdl.predict_cost(ds.samples[pr.slow_idx].features);
        if (pf < ps) model_ok++;
        if (ds.samples[pr.fast_idx].static_cost < ds.samples[pr.slow_idx].static_cost) {
            static_ok++;
        }
    }
    const double n_p = static_cast<double>(std::max<size_t>(1, pair_idxs.size()));
    em.rank_acc = static_cast<float>(static_ok == 0 && pair_idxs.empty() ? 0.0 : static_cast<double>(model_ok) / n_p);
    em.static_rank_acc = static_cast<float>(static_ok == 0 && pair_idxs.empty() ? 0.0 : static_cast<double>(static_ok) / n_p);
    return em;
}

float softplus_f(float z) noexcept {
    if (z > 12.0f) return z;
    if (z < -20.0f) return std::exp(-20.0f);
    return std::log1pf(std::exp(z));
}

float sigmoid_f(float z) noexcept {
    if (z >= 0.0f) {
        const float e = std::exp(-z);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(z);
    return e / (1.0f + e);
}

} // namespace
} // namespace brotensor

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    try {
    using namespace brotensor;
    brotensor::init();
    const TrainConfig cfg = parse_args(argc, argv);

    Device dev = Device::CPU;
    if (cfg.device_arg == "cuda") {
        if (!is_available(Device::CUDA)) {
            std::fprintf(stderr, "CUDA requested but unavailable.\n");
            return 1;
        }
        dev = Device::CUDA;
    } else if (cfg.device_arg == "auto") {
        dev = is_available(Device::CUDA) ? Device::CUDA : Device::CPU;
    }
    std::printf("[train] Using brotensor device: %s\n", device_name(dev));

    const auto ds = inst_opt_train::generate_inst_opt_dataset(cfg.seed, true);
    if (ds.train_indices.empty() || ds.val_indices.empty()) {
        std::fprintf(stderr, "Empty train/val dataset.\n");
        return 1;
    }

    InstOptNet net(dev, cfg.seed ^ 0x55AAULL);
    uint64_t rng = cfg.seed;
    int adam_step_idx = 0;

    std::vector<size_t> shuffled_samples = ds.train_indices;
    std::vector<size_t> shuffled_pairs = ds.train_pairs;
    size_t pair_cursor = 0;

    constexpr int D = static_cast<int>(brass::kInstOptInputDim);
    constexpr int NA = static_cast<int>(brass::kNumInstOptActions);
    constexpr float kPi = 3.14159265358979323846f;

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        for (size_t i = shuffled_samples.size(); i > 1; --i) {
            std::swap(shuffled_samples[i - 1], shuffled_samples[lcg(rng) % i]);
        }
        if (!shuffled_pairs.empty()) {
            for (size_t i = shuffled_pairs.size(); i > 1; --i) {
                std::swap(shuffled_pairs[i - 1], shuffled_pairs[lcg(rng) % i]);
            }
        }

        const float prog = static_cast<float>(epoch - 1) / static_cast<float>(std::max(1, cfg.epochs - 1));
        const float lr = 2e-4f + 0.5f * (3e-3f - 2e-4f) * (1.0f + std::cos(kPi * prog));

        double ep_loss_cost = 0.0;
        double ep_loss_rank = 0.0;
        double ep_loss_pol  = 0.0;
        int n_batches = 0;

        for (size_t start = 0; start < shuffled_samples.size(); start += static_cast<size_t>(cfg.batch_size)) {
            const int B_samp = static_cast<int>(std::min<size_t>(cfg.batch_size, shuffled_samples.size() - start));
            const int B_pair_use = shuffled_pairs.empty() ? 0 : std::min<int>(32, static_cast<int>(shuffled_pairs.size()));
            const int B_tot = B_samp + 2 * B_pair_use;

            std::vector<float> host_x(static_cast<size_t>(B_tot) * D, 0.0f);
            for (int i = 0; i < B_samp; ++i) {
                const auto& feat = ds.samples[shuffled_samples[start + static_cast<size_t>(i)]].features;
                std::memcpy(host_x.data() + static_cast<size_t>(i) * D, feat.x.data(), sizeof(float) * D);
            }
            std::vector<size_t> batch_pairs(B_pair_use);
            for (int p = 0; p < B_pair_use; ++p) {
                const size_t pid = shuffled_pairs[pair_cursor % shuffled_pairs.size()];
                pair_cursor++;
                batch_pairs[p] = pid;
                const auto& pr = ds.pairs[pid];
                std::memcpy(host_x.data() + static_cast<size_t>(B_samp + 2 * p) * D,
                            ds.samples[pr.fast_idx].features.x.data(), sizeof(float) * D);
                std::memcpy(host_x.data() + static_cast<size_t>(B_samp + 2 * p + 1) * D,
                            ds.samples[pr.slow_idx].features.x.data(), sizeof(float) * D);
            }

            const TrunkActivations act = net.forward(host_x, B_tot);
            const Tensor zc_cpu = act.Z_cost.to(Device::CPU);
            const Tensor zp_cpu = act.Z_pol.to(Device::CPU);
            const float* zc = zc_cpu.host_f32();
            const float* zp = zp_cpu.host_f32();

            std::vector<float> dz_cost(B_tot, 0.0f);
            std::vector<float> dz_pol(static_cast<size_t>(B_tot) * NA, 0.0f);

            const float inv_bs = 1.0f / static_cast<float>(std::max(1, B_samp));
            for (int i = 0; i < B_samp; ++i) {
                const auto& smp = ds.samples[shuffled_samples[start + static_cast<size_t>(i)]];
                const float z = std::clamp(zc[i], -20.0f, 20.0f);
                const float pred_c = softplus_f(z);
                const float diff_c = pred_c - smp.measured_cost;
                ep_loss_cost += 0.5 * static_cast<double>(diff_c * diff_c) * inv_bs;
                dz_cost[i] = std::clamp(diff_c * sigmoid_f(z) * inv_bs, -2.0f, 2.0f);

                // Softmax cross-entropy on 16 rewrite classes
                const float* row = zp + static_cast<size_t>(i) * NA;
                float mx = -1e30f;
                for (int c = 0; c < NA; ++c) mx = std::max(mx, row[c]);
                float sum_e = 0.0f;
                std::array<float, NA> probs{};
                for (int c = 0; c < NA; ++c) {
                    probs[c] = std::exp(std::clamp(row[c] - mx, -30.0f, 0.0f));
                    sum_e += probs[c];
                }
                const float inv_se = 1.0f / std::max(1e-12f, sum_e);
                for (int c = 0; c < NA; ++c) probs[c] *= inv_se;
                const uint32_t tgt = std::min<uint32_t>(smp.action_label, NA - 1);
                ep_loss_pol += -std::log(std::max(1e-7f, probs[tgt])) * static_cast<double>(inv_bs);
                for (int c = 0; c < NA; ++c) {
                    const float g = (probs[c] - ((static_cast<uint32_t>(c) == tgt) ? 1.0f : 0.0f)) * 0.6f * inv_bs;
                    dz_pol[static_cast<size_t>(i) * NA + c] = g;
                }
            }

            // Pairwise margin ranking loss on equivalent fast vs slow pairs
            const float inv_bp = 1.0f / static_cast<float>(std::max(1, B_pair_use));
            constexpr float kMargin = 0.25f;
            for (int p = 0; p < B_pair_use; ++p) {
                const int ifast = B_samp + 2 * p;
                const int islow = B_samp + 2 * p + 1;
                const float zf = std::clamp(zc[ifast], -20.0f, 20.0f);
                const float zs = std::clamp(zc[islow], -20.0f, 20.0f);
                const float pf = softplus_f(zf);
                const float ps = softplus_f(zs);
                const float viol = kMargin - (ps - pf);
                if (viol > 0.0f) {
                    ep_loss_rank += static_cast<double>(viol) * inv_bp;
                    const float scale = 1.5f * inv_bp;
                    dz_cost[ifast] = std::clamp(scale * sigmoid_f(zf), -2.0f, 2.0f);
                    dz_cost[islow] = std::clamp(-scale * sigmoid_f(zs), -2.0f, 2.0f);
                }
            }

            net.zero_grad();
            net.backward(act, dz_cost, dz_pol, B_tot);
            net.adam_update(lr, ++adam_step_idx);
            n_batches++;
        }

        if (epoch == 1 || epoch % 20 == 0 || epoch == cfg.epochs) {
            const brass::InstOptModel cur_mdl = net.export_brass_model();
            const EvalMetrics val_m = evaluate_split(cur_mdl, ds, ds.val_indices, ds.val_pairs);
            std::printf("[epoch %3d/%d] lr=%.5f cost_loss=%.4f rank_loss=%.4f pol_loss=%.4f | "
                        "VAL cost_mae=%.3f rank_acc=%.1f%% (static=%.1f%%) pol_top1=%.1f%% pol_top3=%.1f%%\n",
                        epoch, cfg.epochs, lr,
                        ep_loss_cost / std::max(1, n_batches),
                        ep_loss_rank / std::max(1, n_batches),
                        ep_loss_pol / std::max(1, n_batches),
                        val_m.cost_mae,
                        val_m.rank_acc * 100.0f,
                        val_m.static_rank_acc * 100.0f,
                        val_m.pol_top1 * 100.0f,
                        val_m.pol_top3 * 100.0f);
        }
    }

    const brass::InstOptModel fp32_model = net.export_brass_model();
    {
        std::error_code ec;
        const auto bp = std::filesystem::path(cfg.out_brim).parent_path();
        if (!bp.empty()) std::filesystem::create_directories(bp, ec);
        const auto ip = std::filesystem::path(cfg.out_inc).parent_path();
        if (!ip.empty()) std::filesystem::create_directories(ip, ec);
    }
    if (!fp32_model.save_file(cfg.out_brim, true)) {
        std::fprintf(stderr, "Failed to save %s\n", cfg.out_brim.c_str());
        return 1;
    }
    if (!fp32_model.export_cpp_inc(cfg.out_inc)) {
        std::fprintf(stderr, "Failed to export %s\n", cfg.out_inc.c_str());
        return 1;
    }

    brass::InstOptModel int8_model;
    if (!brass::InstOptModel::load_file(cfg.out_brim, int8_model)) {
        std::fprintf(stderr, "Failed to reload quantized %s\n", cfg.out_brim.c_str());
        return 1;
    }

    double sum_rel_err = 0.0, max_abs_diff = 0.0;
    for (size_t idx : ds.val_indices) {
        const float pf = fp32_model.predict_cost(ds.samples[idx].features);
        const float pq = int8_model.predict_cost(ds.samples[idx].features);
        const double ad = std::fabs(pf - pq);
        max_abs_diff = std::max(max_abs_diff, ad);
        sum_rel_err += ad / std::max(0.1, static_cast<double>(pf));
    }
    const EvalMetrics val_i8 = evaluate_split(int8_model, ds, ds.val_indices, ds.val_pairs);
    std::printf("[export] Saved INT8 .brim -> %s and C++ builtin -> %s\n",
                cfg.out_brim.c_str(), cfg.out_inc.c_str());
    std::printf("[export] INT8 vs FP32 val mean_rel_err=%.3f%% max_abs_diff=%.4f | "
                "INT8 val rank_acc=%.1f%% (static=%.1f%%) pol_top1=%.1f%%\n",
                100.0 * sum_rel_err / std::max<size_t>(1, ds.val_indices.size()),
                max_abs_diff,
                val_i8.rank_acc * 100.0f,
                val_i8.static_rank_acc * 100.0f,
                val_i8.pol_top1 * 100.0f);
    return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
