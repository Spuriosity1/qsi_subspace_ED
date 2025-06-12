#pragma once

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
typedef __m128i uint128_t;
#else
typedef __uint128_t uint128_t;
#endif


union Uint128 {
    uint128_t uint128;
    uint64_t uint64[2];
    Uint128(){}
    template<typename T=__uint128_t>
	Uint128(T x){
		uint128 = x;
	}
    Uint128(uint64_t x1, uint64_t x0){
        uint64[0] = x0;
        uint64[1] = x1;
    }

    Uint128 operator^=(const Uint128& other){
        uint128 ^= other.uint128;
        return *this;
    }

    bool operator<(const Uint128& other) const {
        return uint128 < other.uint128;
    }
};


// Hash function for Uint128 to use in unordered_map
struct Uint128Hash {
	std::size_t operator()(const Uint128& b) const {
		return std::hash<uint64_t>()(b.uint64[0]) ^ std::hash<uint64_t>()(b.uint64[1]);
	}
};

struct Uint128Eq {
	bool operator()(const Uint128& a, const Uint128& b) const {
		return (a.uint64[0] == b.uint64[0]) && (a.uint64[1] == b.uint64[1]);
	}
};


static inline   int   popcnt_u128 (const Uint128& n)
{
    const int  cnt_hi  = __builtin_popcountll(n.uint64[1]);
    const int  cnt_lo  = __builtin_popcountll(n.uint64[0]);
    const int  cnt     = cnt_hi + cnt_lo;

    return  cnt;
}

template<typename T>
inline void or_bit(Uint128& x, T i){	
    __uint128_t l=1;
	l <<= i;
    x.uint128 |= l;
}

template<typename T>
inline bool readbit(const Uint128&x, T i){
    __uint128_t l=1;
    l <<= i;
    return x.uint128 & l;
}

inline Uint128 operator&(const Uint128& x, const Uint128& y){
    Uint128 res(x);
    res.uint128 &= y.uint128;
    return res;
}

inline Uint128 operator^(const Uint128& x, const Uint128& y){
    Uint128 res(x);
    res.uint128 ^= y.uint128;
    return res;
}


inline bool operator==(const Uint128& x, const Uint128& y){
    return x.uint128 == y.uint128;
}

template <typename T>
inline Uint128 operator>>(Uint128 x, T idx){
    Uint128 res(x);
    res.uint128 = res.uint128 >> idx;
    return res;
}

inline Uint128 operator|(Uint128 x, const Uint128& y){
	x.uint64[0] |= y.uint64[0];
	x.uint64[1] |= y.uint64[1];
	return x;
}
	


template <typename T>
requires std::convertible_to<T, int>
inline Uint128 operator<<(Uint128 x, T idx){
    Uint128 res(x);
    res.uint128 = res.uint128 << idx;
    return res;
}


template <typename T>
inline Uint128 make_mask(T idx){
    // returns all ones up to (but excluding) idx
    Uint128 res;
    res.uint128 = 1;
    res.uint128 <<= idx;
    res.uint128 --;
    return res;
}

