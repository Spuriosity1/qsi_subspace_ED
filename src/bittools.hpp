#pragma once
#include <cstdint>


typedef __uint128_t  Uint128 ;

static inline   int   popcnt_u128 (__uint128_t n)
{
    const uint64_t      n_hi    = n >> 64;
    const uint64_t      n_lo    = n;
    const int  cnt_hi  = __builtin_popcountll(n_hi);
    const int  cnt_lo  = __builtin_popcountll(n_lo);
    const int  cnt     = cnt_hi + cnt_lo;

    return  cnt;
}
