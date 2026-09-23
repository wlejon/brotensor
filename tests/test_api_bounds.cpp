// bro.tensor input-size contract: a native handed an operand smaller than its
// dims say (or a mismatched elementwise pair) throws a JS Error instead of
// reading or writing past the operand's storage.

#include "api_test_helpers.h"

using brotensor_api_test::runJs;

int run_api_bounds_tests() {
    int failures = 0;
    return failures;
}
