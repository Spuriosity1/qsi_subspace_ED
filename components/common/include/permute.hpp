#include "bittools.hpp"
#include <cassert>
#include <numeric>
#include <stdint.h>
#include <array>


#if defined(__x86_64__)

// Source - https://stackoverflow.com/a/76469644
// Posted by Ovinus Real
// Retrieved 2025-11-19, License - CC BY-SA 4.0

void permute_128(const char* in, char* out,
    __m512i byte_idx1, __m512i bit_mask1,
    __m512i byte_idx2, __m512i bit_mask2) {
    __m512i in_v = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) in));

    __mmask64 permuted_1 = _mm512_test_epi8_mask(
        _mm512_shuffle_epi8(in_v, byte_idx1), bit_mask1);
    __mmask64 permuted_2 = _mm512_test_epi8_mask(
        _mm512_shuffle_epi8(in_v, byte_idx2), bit_mask2);

    _store_mask64((__mmask64*) out, permuted_1);
    asm volatile ("" ::: "memory");
    _store_mask64((__mmask64*) out + 1, permuted_2);
}

void permute_128_array(char* arr, size_t count, uint8_t idx[128]) {
    __m512i idx1 = _mm512_loadu_si512(idx);
    __m512i idx2 = _mm512_loadu_si512(idx + 64);

    __m512i byte_idx1, bit_mask1, byte_idx2, bit_mask2;
    get_permute_constants(idx1, &byte_idx1, &bit_mask1);
    get_permute_constants(idx2, &byte_idx2, &bit_mask2);

    count *= 16;

    for (size_t i = 0; i < count; i += 16)
        permute_128(arr + i, arr + i, byte_idx1, bit_mask1, byte_idx2, bit_mask2);
}

class Permute128 {
private:
    __m512i byte_idx1, bit_mask1;
    __m512i byte_idx2, bit_mask2;
    
    // Helper function to compute permutation constants
    static void get_permute_constants(__m512i idx, __m512i* byte_idx, __m512i* bit_mask) {
        // Each index tells us which bit to extract
        // Divide by 8 to get byte index, modulo 8 to get bit position
        __m512i div8 = _mm512_srli_epi8(idx, 3);  // idx / 8
        __m512i mod8 = _mm512_and_si512(idx, _mm512_set1_epi8(7));  // idx % 8
        
        *byte_idx = div8;
        
        // Create bit mask: 1 << (idx % 8)
        __m512i one = _mm512_set1_epi8(1);
        *bit_mask = _mm512_sllv_epi8(one, mod8);
    }
    
    // Store 64-bit mask to memory
    static inline void _store_mask64(__mmask64* ptr, __mmask64 mask) {
        *ptr = mask;
    }
    
    // Core permutation function
    void permute_internal(const char* in, char* out) const {
        __m512i in_v = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) in));
        __mmask64 permuted_1 = _mm512_test_epi8_mask(
            _mm512_shuffle_epi8(in_v, byte_idx1), bit_mask1);
        __mmask64 permuted_2 = _mm512_test_epi8_mask(
            _mm512_shuffle_epi8(in_v, byte_idx2), bit_mask2);
        _store_mask64((__mmask64*) out, permuted_1);
        asm volatile ("" ::: "memory");
        _store_mask64((__mmask64*) out + 1, permuted_2);
    }

public:
    // Constructor accepts permutation indices
    explicit Permute128(const std::vector<uint8_t>& indices) {
        if (indices.size() != 128) {
            throw std::invalid_argument("Permutation must have exactly 128 indices");
        }
        
        // Validate indices are in range [0, 127]
        for (uint8_t idx : indices) {
            if (idx >= 128) {
                throw std::invalid_argument("All indices must be in range [0, 127]");
            }
        }
        
        // Load indices into AVX-512 registers
        __m512i idx1 = _mm512_loadu_si512(indices.data());
        __m512i idx2 = _mm512_loadu_si512(indices.data() + 64);
        
        // Precompute permutation constants
        get_permute_constants(idx1, &byte_idx1, &bit_mask1);
        get_permute_constants(idx2, &byte_idx2, &bit_mask2);
    }
    
    // Fast in-place permutation
    __uint128_t& permute_in_place(__uint128_t& x) const {
        permute_internal(reinterpret_cast<char*>(&x), reinterpret_cast<char*>(&x));
        return x;
    }
    
    // Non-modifying permutation
    __uint128_t permute(__uint128_t x) const {
        __uint128_t result;
        permute_internal(reinterpret_cast<const char*>(&x), reinterpret_cast<char*>(&result));
        return result;
    }
};

/*
#elif defined(__ARM_NEON)
#include <arm_neon.h>

TODO if anyone can be bothered: ARM implementation
*/
#else
// fallback implementation

class Permute128 {
private:

    std::array<uint8_t, 128> I;
    void permute_internal(const __uint128_t src, __uint128_t& dest) const{
        dest = 0;
        for (uint8_t n = 0; n < I.size(); n++) {
            uint8_t to = static_cast<uint8_t>(I[n]);
            dest |= ((src >> n) & 1) << to;
        }
    }


public:
    // Constructor accepts permutation indices
    explicit Permute128(const std::vector<uint8_t>& indices) {
        // Validate indices are in range [0, 127]
        for (uint8_t idx : indices) {
            if (idx >= 128) {
                throw std::invalid_argument("All indices must be in range [0, 127]");
            }
        }

        for (int i=0; i<128; i++){
            if (i < indices.size()){
                I[i] = indices[i];
            } else {
                I[i] = i;
            }
        }
    }
    
    // Fast in-place permutation 
    __uint128_t& permute_in_place(__uint128_t& x) const {
        __uint128_t tmp =0;
        permute_internal(x, tmp);
        x = tmp;
        return x;
    }
    
    // Non-modifying permutation
    __uint128_t permute(__uint128_t x) const {
        permute_internal(x, x);
        return x;
    }
};

#endif


