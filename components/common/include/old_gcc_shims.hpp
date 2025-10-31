#pragma once

// Fallback for compilers without std::bit_width
#if !defined(__cpp_lib_bitops) || __cpp_lib_bitops < 201907L
namespace std {
    template<typename T>
    constexpr int bit_width(T x) noexcept {
        if (x == 0) return 0;
        int width = 0;
        while (x != 0) {
            x >>= 1;
            ++width;
        }
        return width;
    }
}
#endif
