#pragma once

#include <string>
#if __cplusplus >= 202002L
    // C++20
    
template<typename T>
concept RealOrCplx = std::floating_point<T> ||
                 (requires { typename T::value_type; } &&
                  std::is_same_v<T, std::complex<typename T::value_type>> &&
                  std::floating_point<typename T::value_type>);



template<typename T>
concept Basis =  std::derived_from<T, ZBasisBase> &&
requires (T b, const ZBasisBase::state_t& state, ZBasisBase::idx_t& J) {
    { b.search(state, J) } -> std::same_as<int>;
};

#else
     // shim for C++17
    #define RealOrCplx typename
    #define Basis typename
#endif


inline std::string format_as_memory(size_t bytes){
    size_t factor = 1024ull*1024*1024*1024;
    if (bytes > factor) { return std::to_string(bytes / factor) + "TB"; }
    factor /= 1024;
    if (bytes > factor) { return std::to_string(bytes / factor) + "GB"; }
    factor /= 1024;
    if (bytes > factor) { return std::to_string(bytes / factor) + "MB"; }
    factor /= 1024;
    if (bytes > factor) { return std::to_string(bytes / factor) + "kB"; }
     return std::to_string(bytes ) + " B"; 
}
