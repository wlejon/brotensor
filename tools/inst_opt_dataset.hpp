#pragma once

#include <brass/mir/inst_opt_model.hpp>
#include <cstdint>
#include <vector>

namespace brotensor::inst_opt_train {

struct InstOptSample {
    brass::InstOptFeatures features;
    float measured_cost = 1.0f;
    uint32_t static_cost = 1;
    uint32_t action_label = 0; // InstOptAction (0..15)
    uint32_t group_id = 0;
    bool is_gpu = false;
};

struct InstOptPair {
    size_t fast_idx = 0;
    size_t slow_idx = 0;
    uint32_t group_id = 0;
};

struct InstOptDataset {
    std::vector<InstOptSample> samples;
    std::vector<InstOptPair> pairs;
    std::vector<size_t> train_indices;
    std::vector<size_t> val_indices;
    std::vector<size_t> train_pairs;
    std::vector<size_t> val_pairs;
    uint32_t num_groups = 0;
    uint32_t x64_benchmarked = 0;
    uint32_t ptx_estimated = 0;
};

/// Generates a real-hardware-benchmarked (x64 via Brass JIT + __rdtsc) and PTX-lowered
/// instruction optimization dataset of verified equivalent MIR expressions.
InstOptDataset generate_inst_opt_dataset(uint64_t seed = 0xB4A55101ULL, bool verbose = true);

} // namespace brotensor::inst_opt_train
