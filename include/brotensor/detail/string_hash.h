#pragma once

// Transparent hash for std::unordered_map<std::string, ...> keys, so lookups
// by std::string_view / const char* don't have to materialize a std::string
// just to satisfy the map's key type. Pair with std::equal_to<> (already
// transparent) as the KeyEqual template argument.

#include <cstddef>
#include <string>
#include <string_view>

namespace brotensor::detail {

struct StringHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }
    std::size_t operator()(const std::string& s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
    std::size_t operator()(const char* s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
};

}  // namespace brotensor::detail
