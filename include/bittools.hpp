#pragma once
#include <cstdint>
#include <cstdio>
#include <string>

union Uint128 {
    __uint128_t uint128;
    uint64_t uint64[2];
    Uint128(){}
    template<typename T>
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

static inline   int   popcnt_u128 (const Uint128& n)
{
    const int  cnt_hi  = __builtin_popcountll(n.uint64[1]);
    const int  cnt_lo  = __builtin_popcountll(n.uint64[0]);
    const int  cnt     = cnt_hi + cnt_lo;

    return  cnt;
}

template<typename T>
inline void or_bit(Uint128& x, T i, __uint128_t l=1){
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


template <typename T>
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

