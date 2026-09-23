// native_tensor_extra.cpp — the brotensor ops that never had a JS binding.
//
// Everything here is a free function on `bro.tensor`; js/tensor_extra.js is
// the wrapper half that validates arguments and reads the error slot back.

#include "native_tensor_extra_decl.h"
#include "api_internal.h"
#include "native_register.h"

#include <string>

using namespace brotensor::api;

extern "C" {

} // extern "C"

namespace brotensor::api {

bool registerTensorNatives_extra(std::string* error) {
    (void)error;
    return true;
}

} // namespace brotensor::api
