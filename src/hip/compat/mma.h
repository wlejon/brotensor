#pragma once

#if defined(__has_include)
#if __has_include(<rocwmma/rocwmma.hpp>)
#include <rocwmma/rocwmma.hpp>
namespace nvcuda {
namespace wmma {
    using namespace ::rocwmma;
}
}
#endif
#endif
