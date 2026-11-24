#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>

namespace brotensor::kernels {

inline constexpr std::uint32_t kSelectBlock = 256;

// SEL_SAMPLE: (a logits, b probs, d indices, e counter_ptr, key_lo, key_hi,
//              V, ctr_lo, ctr_hi, top_k, rows, T, top_p)
brass::Function* build_select_sample_mir(brass::Module& mod, bool has_counter);

// SEL_COUNTER: (e counter_ptr, rows)
brass::Function* build_select_counter_mir(brass::Module& mod);

// SEL_MD: (a logits, b tokens, c lp, d pred, e scores, f conf,
//          key_lo, key_hi, T, C, V, mask_id, k_keep, guidance, layer_penalty, pos_t, class_t)
brass::Function* build_select_md_mir(brass::Module& mod, bool guided, bool has_class_temp, bool has_pos_temp);

// SEL_COMMIT: (a pred, b idx, c tokens, d unmask_step, k, n, step)
brass::Function* build_select_commit_mir(brass::Module& mod);

}  // namespace brotensor::kernels
