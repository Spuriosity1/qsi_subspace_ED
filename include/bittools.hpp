#pragma once

#include <concepts>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <string>


union Uint128 {
    __uint128_t uint128;
    uint64_t uint64[2];
    constexpr Uint128(){uint128 = 0;}

    template<typename T=__uint128_t>
    requires std::convertible_to<T, __uint128_t>
	constexpr Uint128(T x){
    uint128 = x;
	}

    constexpr Uint128(uint64_t x1, uint64_t x0){
        uint64[0] = x0;
        uint64[1] = x1;
    }

    constexpr Uint128 operator^=(const Uint128& other){
        uint128 ^= other.uint128;
        return *this;
    }

    constexpr Uint128 operator&=(const Uint128& other){
        uint128 &= other.uint128;
        return *this;
    }

    constexpr Uint128 operator|=(const Uint128& other){
        uint128 |= other.uint128;
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

    constexpr bool operator>(const Uint128& other) const {
        return uint128 > other.uint128;
    }
    
    template <typename H>
    friend H AbslHashValue(H h, const Uint128& c) {
        return H::combine(std::move(h), c.uint128);
        //return H::combine(std::move(h), c.uint64[0], c.uint64[1]);
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

template <typename T>
constexpr inline Uint128 operator>>(Uint128 x, T idx){
    Uint128 res(x);
    res.uint128 = res.uint128 >> idx;
    return res;
}

constexpr inline Uint128 operator|(Uint128 x, const Uint128& y){
	x.uint64[0] |= y.uint64[0];
	x.uint64[1] |= y.uint64[1];
	return x;
}
	


template <typename T>
requires std::convertible_to<T, int>
constexpr inline Uint128 operator<<(Uint128 x, T idx){
    Uint128 res(x);
    res.uint128 = res.uint128 << idx;
    return res;
}


template <typename T>
inline Uint128 make_mask(T idx){
    // returns all ones up to (but excluding) idx
	// DOES NOT work for idx >= 128 (deliberate omission -- avoid branches)
    Uint128 res;
    res.uint128 = 1;
    res.uint128 <<= idx;
    res.uint128--;
    return res;
}

template <typename T>
requires std::convertible_to<T, size_t>
inline Uint128 permute(const Uint128& x, const std::vector<T>& I) {
    // Applies the permutation I to the bits of x
    // such that y & (1 << I[n]) == x & (1 << n)
    Uint128 y = 0;
    for (size_t n = 0; n < I.size(); n++) {
        size_t to = static_cast<size_t>(I[n]);
        y |= ((x >> n) & 1) << to;
    }
    return y;
}
