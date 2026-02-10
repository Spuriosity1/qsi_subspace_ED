#pragma once

#include <iomanip>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
// #include <bit>
//
#include <cstdint>
#include <climits>


/*
#if defined(__INTEL_LLVM_COMPILER)
#include <concepts>
#endif

constexpr int bit_width(uint64_t x) noexcept {
    if (x == 0) return 0;
    
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
    // Use compiler intrinsics for best performance
    return 64 - __builtin_clzll(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse64(&idx, x);
    return idx + 1;
#else
    #warning "using untested, manual bit_width code!"
    // Portable fallback using De Bruijn multiplication
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    
    static constexpr int debruijn_table[64] = {
        0,  47,  1, 56, 48, 27,  2, 60,
       57, 49, 41, 37, 28, 16,  3, 61,
       54, 58, 35, 52, 50, 42, 21, 44,
       38, 32, 29, 23, 17, 11,  4, 62,
       46, 55, 26, 59, 40, 36, 15, 53,
       34, 51, 20, 43, 31, 22, 10, 45,
       25, 39, 14, 33, 19, 30,  9, 24,
       13, 18,  8, 12,  7,  6,  5, 63
    };
    
    return debruijn_table[(x * 0x03f79d71b4cb0a89ULL) >> 58] + 1;
#endif
}

// Overloads for other unsigned types
constexpr int bit_width(uint32_t x) noexcept {
    if (x == 0) return 0;
#if defined(__GNUC__) || defined(__clang__)
    return 32 - __builtin_clz(x);
#else
    return bit_width(uint64_t(x));
#endif
}

*/


union Uint128 {
    static const int i64_width=2;
    __uint128_t uint128=0;
    uint64_t uint64[2];
    constexpr Uint128(){uint128 = 0;}

	//Uint128(uint64_t x){ this->uint128 = x;}
	Uint128(__uint128_t x){ this->uint128 = x;}

    constexpr Uint128(uint64_t x1, uint64_t x0){
        uint64[0] = x0;
        uint64[1] = x1;
    }

    constexpr Uint128& operator^=(const Uint128& other){
        uint128 ^= other.uint128;
        return *this;
    }

    constexpr Uint128& operator&=(const Uint128& other){
        uint128 &= other.uint128;
        return *this;
    }

    constexpr Uint128& operator|=(const Uint128& other){
        uint128 |= other.uint128;
        return *this;
    }
    
    template <typename T>
    constexpr Uint128& operator<<=(T x){
        uint128 <<= x;
        return *this;
    }

    template <typename T>
    constexpr Uint128& operator>>=(T other){
        uint128 >>= other;
        return *this;
    }


    constexpr Uint128 operator~() const {
        Uint128 x = *this;
        x.uint64[0] = ~x.uint64[0];
        x.uint64[1] = ~x.uint64[1];
        return x;
    }

    constexpr bool operator<(const Uint128& other) const {
        return uint128 < other.uint128;
    }


    constexpr bool operator<=(const Uint128& other) const {
        return uint128 <= other.uint128;
    }

    constexpr bool operator>(const Uint128& other) const {
        return uint128 > other.uint128;
    }

    constexpr bool operator>=(const Uint128& other) const {
        return uint128 >= other.uint128;
    }
    
    template <typename H>
    friend H AbslHashValue(H h, const Uint128& c) {
        return H::combine(std::move(h), c.uint128);
        //return H::combine(std::move(h), c.uint64[0], c.uint64[1]);
    }

    constexpr uint64_t bit_width() const {
        return 64*i64_width;
    }

};


// Hash function for Uint128 to use in unordered_map
struct Uint128Hash {
	std::size_t operator()(const Uint128& b) const {
		return std::hash<uint64_t>()(b.uint64[0]) ^ (std::hash<uint64_t>()(b.uint64[1]));
	}
};

struct Uint128Eq {
	bool operator()(const Uint128& a, const Uint128& b) const {
		return (a.uint64[0] == b.uint64[0]) && (a.uint64[1] == b.uint64[1]);
	}
};


// Specialization for std::hash
namespace std {
    template<>
    struct hash<Uint128> {
        std::size_t operator()(const Uint128& b) const {
            return Uint128Hash()(b);
        }
    };
}


constexpr inline   int   popcnt_u128 (const Uint128& n)
{
    const int  cnt_hi  = __builtin_popcountll(n.uint64[1]);
    const int  cnt_lo  = __builtin_popcountll(n.uint64[0]);
    const int  cnt     = cnt_hi + cnt_lo;

    return  cnt;
}

template<typename T>
constexpr inline void or_bit(Uint128& x, T i){	
    __uint128_t l=1;
	l <<= i;
    x.uint128 |= l;
}


template<typename T>
constexpr inline void xor_bit(Uint128& x, T i){	
    __uint128_t l=1;
	l <<= i;
    x.uint128 ^= l;
}

template<typename T>
constexpr inline bool readbit(const Uint128&x, T i){
    __uint128_t l=1;
    l <<= i;
    return x.uint128 & l;
}

constexpr inline Uint128 operator&(const Uint128& x, const Uint128& y){
    Uint128 res(x);
    res &= y;
    return res;
}

constexpr inline Uint128 operator^(const Uint128& x, const Uint128& y){
    Uint128 res(x);
    res ^= y;
    return res;
}


constexpr inline bool operator==(const Uint128& x, const Uint128& y){
    return x.uint128 == y.uint128;
}


constexpr inline bool operator!=(const Uint128& x, const Uint128& y){
    return x.uint128 != y.uint128;
}

constexpr inline Uint128 operator|(Uint128 x, const Uint128& y){
	x.uint64[0] |= y.uint64[0];
	x.uint64[1] |= y.uint64[1];
	return x;
}
	

template <typename T>
//requires std::convertible_to<T, int>
constexpr inline std::enable_if_t<std::is_integral_v<T>, Uint128> 
operator>>(Uint128 x, T idx){
    Uint128 res(x);
    res.uint128 = res.uint128 >> idx;
    return res;
}

template <typename T>
//requires std::convertible_to<T, int>
constexpr inline std::enable_if_t<std::is_integral_v<T>, Uint128> 
operator<<(Uint128 x, T idx){
    Uint128 res(x);
    res.uint128 = res.uint128 << idx;
    return res;
}


inline int highest_set_bit(const Uint128& x) {
    if (x == 0) return -1;

    if (x.uint64[1] != 0) {
        return 64 + 63 - __builtin_clzll(x.uint64[1]);
    } else {
        return 63 - __builtin_clzll(x.uint64[0]);
    }
}

inline int lowest_set_bit(const Uint128& x) {
    if (x == 0) return 128;
    if (x.uint64[0] != 0) {
        return __builtin_ctzll(x.uint64[0]);
    } else {
        return 64 + __builtin_ctzll(x.uint64[1]);
    }
}


template <typename T>
inline Uint128 make_mask(T idx){
    // returns all ones up to (but excluding) idx
	// DOES NOT work for idx >= 128 (deliberate omission -- avoid branches)
    // in other words, returns [idx] 1's from the LSB
    Uint128 res;
    res.uint128 = 1;
    res.uint128 <<= idx;
    res.uint128--;
    return res;
}

template <typename UintT, typename T>
//requires std::convertible_to<T, uint8_t>
inline UintT permute(const UintT& x, const std::vector<T>& I) {
    // Applies the permutation I to the bits of x
    // such that y & (1 << I[n]) == x & (1 << n)
    UintT y = 0;
    for (uint8_t n = 0; n < I.size(); n++) {
        uint8_t to = static_cast<uint8_t>(I[n]);
        y |= ((x >> n) & 1) << to;
    }
    return y;
}


// Alternative: Hexadecimal output operator
inline std::ostream& printHex(std::ostream& os, const Uint128& val) {
    std::ios_base::fmtflags flags = os.flags();
    
    os << "0x" << std::hex << std::setfill('0');
    
    // Print high 64 bits if non-zero
    if (val.uint64[1] != 0) {
        os << val.uint64[1];
        os << std::setw(16) << val.uint64[0];
    } else {
        os << val.uint64[0];
    }
    
    os.flags(flags);
    return os;
}


inline std::ostream& operator<<(std::ostream& os, const Uint128& val) {
    std::ios_base::fmtflags flags = os.flags();
    
    os << "0x" << std::hex << std::setfill('0');
    
    // Print high 64 bits if non-zero
    if (val.uint64[1] != 0) {
        os << val.uint64[1];
        os << std::setw(16) << val.uint64[0];
    } else {
        os << val.uint64[0];
    }
    
    os.flags(flags);
    return os;
}

inline std::string to_string(const Uint128& x){
    std::stringstream oss;
    printHex(oss, x);
    return oss.str();
}


template <typename T>
std::ostream& printvec(std::ostream& os, const std::vector<T>& v){
    for (const auto& x : v){
        os << x<<", ";
    }
    return os;
}

