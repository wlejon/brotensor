#pragma once

#include <functional>
#include <string>
#include "embed/embed.h"
#include "brotensor/tensor.h"

namespace brotensor::api {

/// Registers native tensor functions and mounts the tensor JS module into the current Bronze realm.
void installTensor();

/// How a path handed to `bro.tensor.openSafetensors` / `saveSafetensors`
/// becomes a filesystem path. Unset, the path is used as given; a host sets
/// its `fs` resolver so a relative path means what it means to the app
/// (bro: `setPathResolver(&brokit::api::resolveAssetPath)` before
/// installTensor, the way it does for bromesh and brosoundml). Process-wide:
/// every realm shares the host's filesystem.
void setPathResolver(std::function<std::string(const std::string&)> resolver);

// ---- Access -----------------------------------------------------------------
//
// Which files a program may touch is the host's decision, not brotensor's:
// each native handed a path asks the host's one access gate,
// bronze::embed::checkAccess (embed/embed_access.h), with the path resolved
// (setPathResolver), absolute and lexically normal (UTF-8), on the JS thread
// that made the call, before the file is opened. With no gate installed every
// path is allowed and nothing changes.
//
//   openSafetensors   the path Read (the tensors are read later from that mapping)
//   saveSafetensors   the path Write (asked before any tensor is downloaded)
//
// A refusal is bronze::embed::newAccessError, thrown by js/tensor.js (the
// native returns its failure and __bro_native.tensor.takeAccessError hands the
// Error over): an Error whose `message` is the gate's sentence (named
// PermissionRevoked for a grant taken back), `code` is "EACCES" and `path` the
// absolute path asked about. Nothing is read or written.

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
