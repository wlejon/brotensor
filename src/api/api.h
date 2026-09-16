#pragma once

#include <string>
#include "embed/embed.h"
#include "brotensor/tensor.h"

namespace brotensor::api {

/// Registers native tensor functions and mounts the tensor JS module into the current Bronze realm.
void installTensor();

/// Register all __bro_native.tensor.* native function bindings with Bronze.
bool registerTensorNatives(std::string* error = nullptr);

/// Mount the compiled JS tensor module (bronze_tensor_main) into the current realm.
void installTensorJS();

/// Retrieve the underlying brotensor::Tensor pointer from a Bronze Value handle (GpuTensor).
/// Returns nullptr if val is not a valid GpuTensor handle.
brotensor::Tensor* getTensorFromHandle(bronze::Value val);

/// Wrap a brotensor::Tensor into a Bronze GpuTensor Value handle.
bronze::Value createGpuTensorValue(brotensor::Tensor tensor);

} // namespace brotensor::api
