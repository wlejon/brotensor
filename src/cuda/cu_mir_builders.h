#pragma once

#include "cu_mir_cache.h"

#if BROTENSOR_HAS_BRASS_CUDA_JIT
#include <brass/mir/module.hpp>
#include <brass/target/ptx_target.hpp>
#endif

namespace brotensor::detail::cuda {

#if BROTENSOR_HAS_BRASS_CUDA_JIT
void configure_mir_ptx_options(const MirKey& k, brass::target::PtxOptions& opts);
brass::Function* build_mir_kernel(brass::Module& mod, const MirKey& k);
#endif

}  // namespace brotensor::detail::cuda
