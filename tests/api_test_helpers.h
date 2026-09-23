#pragma once

// Shared harness for the bro.tensor JS API tests.
//
// test_api.cpp owns main() and the realm; each restored group keeps its own
// checks in tests/test_api_<group>.cpp, exposing one
// `int run_api_<group>_tests()` that answers the number of failed assertions.
// main() sums them. The group files link into the same brotensor_test_api
// executable, so the realm is installed exactly once.

#include "embed/embed.h"
#include "eval/eval.h"

#include <cstdio>
#include <string>

namespace brotensor_api_test {

// Run one JS block; a throw or a non-"OK" result is one failure.
inline int runJs(const char* name, const std::string& script) {
    std::printf("%s...\n", name);
    auto r = bronze::eval::evalScript(script);
    if (r.thrown) {
        std::printf("  JS Exception thrown: %s\n", bronze::embed::toUtf8(r.value).c_str());
        return 1;
    }
    const std::string out = bronze::embed::toUtf8(r.value);
    if (out != "OK") {
        std::printf("  unexpected result: %s\n", out.c_str());
        return 1;
    }
    return 0;
}

} // namespace brotensor_api_test

// Every restored group's entry point (defined in tests/test_api_<group>.cpp).
int run_api_batched_tests();
int run_api_attn2_tests();
int run_api_audio_tests();
int run_api_conv_tests();
int run_api_int8_tests();
int run_api_misc_tests();
int run_api_extra_tests();
int run_api_bounds_tests();
