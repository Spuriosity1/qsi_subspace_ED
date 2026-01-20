#include "bittools.hpp"
#include <cassert>
#include <numeric>
#include <stdint.h>
#include <array>



#if defined(__AVX512F__)
#include <immintrin.h>

//#pragma clang attribute push(__attribute__((target("avx512f,avx512bw"))), apply_to=function)

// Source - https://stackoverflow.com/a/76469644
// Posted by Ovinus Real
// Retrieved 2025-11-19, License - CC BY-SA 4.0

static const __m512i bit_idx_mask = _mm512_set1_epi8(0x7);
static const __m512i bit_mask_lookup = _mm512_set1_epi64(0x8040201008040201);


class Permute128_AVX512 {
private:
    __m512i byte_idx1, bit_mask1;
    __m512i byte_idx2, bit_mask2;


    static void get_permute_constants(__m512i idx, __m512i* byte_idx, __m512i* bit_mask) {
        *byte_idx = _mm512_srli_epi32(_mm512_andnot_si512(bit_idx_mask, idx), 3);
        idx = _mm512_and_si512(idx, bit_idx_mask);
        *bit_mask = _mm512_shuffle_epi8(bit_mask_lookup, idx);
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
    explicit Permute128_AVX512(const std::vector<uint8_t>& indices) {
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


#endif


#if defined(__AVX2__)
#include <immintrin.h>


class Permute128_AVX2 {
private:
    uint8_t pos[128];
    static uint32_t get_32_128_bits(__m256i x, __m256i pos) {                           /* extract 32 permuted bits out from 2x128 bits   */
        __m256i pshufb_mask  = _mm256_set_epi8(0,0,0,0, 0,0,0,0, 128,64,32,16, 8,4,2,1, 0,0,0,0, 0,0,0,0, 128,64,32,16, 8,4,2,1);
        __m256i byte_pos     = _mm256_srli_epi32(pos, 3);                       /* which byte do we need within the 16 byte lanes. bits 6,5,4,3 select the right byte */
        byte_pos     = _mm256_and_si256(byte_pos, _mm256_set1_epi8(0xF)); /* mask off the unwanted bits (unnecessary if _mm256_srli_epi8 would have existed   */
        __m256i bit_pos      = _mm256_and_si256(pos, _mm256_set1_epi8(0x07));   /* which bit within the byte                 */
        __m256i bit_pos_mask = _mm256_shuffle_epi8(pshufb_mask, bit_pos);       /* get bit mask                              */

        __m256i bytes_wanted = _mm256_shuffle_epi8(x, byte_pos);                /* get the right bytes                       */
        __m256i bits_wanted  = _mm256_and_si256(bit_pos_mask, bytes_wanted);    /* apply the bit mask to get rid of the unwanted bits within the byte */
        __m256i bits_x8      = _mm256_cmpeq_epi8(bits_wanted, bit_pos_mask);    /* set all bits if the wanted bit is set     */
        return _mm256_movemask_epi8(bits_x8);                           /* move most significant bit of each byte to 32 bit register */
    }



    // Core permutation function
    void permute_internal(const char* in, char* out) const {

        uint64_t t0, t1, t2, t3, t10, t32;
        __m256i  x2 = _mm256_broadcastsi128_si256(*reinterpret_cast<const __m128i*>(in));   /* broadcast x to the hi and lo lane                            */
        t0 = get_32_128_bits(x2, _mm256_loadu_si256((__m256i*)&pos[0]));
        t1 = get_32_128_bits(x2, _mm256_loadu_si256((__m256i*)&pos[32]));
        t2 = get_32_128_bits(x2, _mm256_loadu_si256((__m256i*)&pos[64]));
        t3 = get_32_128_bits(x2, _mm256_loadu_si256((__m256i*)&pos[96]));
        t10 = (t1<<32)|t0;
        t32 = (t3<<32)|t2;
        *reinterpret_cast<__m128i*>(out) = _mm_set_epi64x(t32, t10);
    }

public:
    // Constructor accepts permutation indices
    explicit Permute128_AVX2(const std::vector<uint8_t>& indices) {
        if (indices.size() != 128) {
            throw std::invalid_argument("Permutation must have exactly 16 indices (for 128 bits)");
        }
        for (int i=0; i<128; i++){
            pos[ indices[i]] = i;
        }
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


#endif
/*
#elif defined(__ARM_NEON)
#include <arm_neon.h>

TODO if anyone can be bothered: ARM implementation
*/

// fallback implementation

class Permute128_fallback {
private:

    struct BitExtract {
        uint8_t src_byte;   // 0..15
        uint8_t src_bit;    // 0..7
        uint8_t dst_bit;    // 0..7
    };

    // For each output byte, we store exactly 8 bit extraction operations
    BitExtract compiled_ops[16][8];


    std::array<uint8_t, 128> I;
    void permute_internal(const __uint128_t _src, __uint128_t& _dest) const{
        /*
        dest = 0;
        for (uint8_t n = 0; n < I.size(); n++) {
            uint8_t to = static_cast<uint8_t>(I[n]);
            dest |= ((src >> n) & 1) << to;
        }
        */
        auto src = reinterpret_cast<const char*>(&_src);
        auto dst = reinterpret_cast<char*>(&_dest);

        for (int ob = 0; ob < 16; ob++) {
            uint8_t acc = 0;

            // Combine 8 bit extractions for this output byte
            const BitExtract* ops = compiled_ops[ob];

            acc |= ((src[ops[0].src_byte] >> ops[0].src_bit) & 1u) << ops[0].dst_bit;
            acc |= ((src[ops[1].src_byte] >> ops[1].src_bit) & 1u) << ops[1].dst_bit;
            acc |= ((src[ops[2].src_byte] >> ops[2].src_bit) & 1u) << ops[2].dst_bit;
            acc |= ((src[ops[3].src_byte] >> ops[3].src_bit) & 1u) << ops[3].dst_bit;
            acc |= ((src[ops[4].src_byte] >> ops[4].src_bit) & 1u) << ops[4].dst_bit;
            acc |= ((src[ops[5].src_byte] >> ops[5].src_bit) & 1u) << ops[5].dst_bit;
            acc |= ((src[ops[6].src_byte] >> ops[6].src_bit) & 1u) << ops[6].dst_bit;
            acc |= ((src[ops[7].src_byte] >> ops[7].src_bit) & 1u) << ops[7].dst_bit;

            dst[ob] = acc;
        }
    }
    

public:
    // Constructor accepts permutation indices
    explicit Permute128_fallback(const std::vector<uint8_t>& indices) {
        // Validate indices are in range [0, 127]
        for (uint8_t idx : indices) {
            if (idx >= 128) {
                throw std::invalid_argument("All indices must be in range [0, 127]");
            }
        }

        for (int i=0; i<128; i++){
            if (static_cast<size_t>(i) < indices.size()){
                I[indices[i]] = i;
            } else {
                I[i] = i;
            }
        }

        for (int ob = 0; ob < 16; ob++) {
            for (int bit = 0; bit < 8; bit++) {

                int out_bit_index = ob * 8 + bit;
                int in_bit_index  = I[out_bit_index];

                compiled_ops[ob][bit] = BitExtract{
                    uint8_t(in_bit_index >> 3),   // source byte
                    uint8_t(in_bit_index & 7),    // source bit
                    uint8_t(bit)                  // destination bit
                };
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



#if defined(__AVX512F__)
using Permute128 = Permute128_AVX512;
#elif defined(__AVX2__)
using Permute128 = Permute128_AVX2;
#else
using Permute128 = Permute128_fallback;
#endif

