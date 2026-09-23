#pragma once

// native_tensor_extra_decl.h — the brotensor ops that had no JS binding at all
// (neither in the QuickJS binding nor in the bronze port). Bodies in
// native_tensor_extra.cpp, wrappers in js/tensor_extra.js.

#include <stdbool.h>
#include <stdint.h>
#include "abi/bronze_native_type.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
namespace brotensor::api { bool registerTensorNatives_extra(std::string* error); }
#endif
