#include "bittools.hpp"
#include "permute.hpp"
#include <algorithm>
#include <random>

using UintT = Uint128;

// Naive reference: moves input bit n to output bit perm[n],
// matching the semantics of bittools::permute().
static __uint128_t naive_permute(const __uint128_t x, const std::vector<uint8_t>& perm) {
    __uint128_t y = 0;
    for (int n = 0; n < 128; n++) {
        __uint128_t bit = (x >> n) & 1;
        y |= bit << perm[n];
    }
    return y;
}

// -----------------------------------------------------------------------

template<typename Perm128>
static bool check_one(const Perm128& p,
                      const __uint128_t input,
                      const std::vector<uint8_t>& perm,
                      const char* label)
{
    __uint128_t expected = naive_permute(input, perm);

    // Test non-modifying permute()
    __uint128_t got = p.permute(input);
    if (got != expected) {
        std::cout << "  FAIL [" << label << "] permute() mismatch\n";
        return false;
    }

    // Test permute_in_place(): must give the same answer and not corrupt
    __uint128_t x = input;
    p.permute_in_place(x);
    if (x != expected) {
        std::cout << "  FAIL [" << label << "] permute_in_place() mismatch\n";
        return false;
    }

    return true;
}

template<typename Perm128>
static bool test_one_perm(const std::vector<uint8_t>& perm,
                          const std::string& test_name)
{
    Perm128 p(perm);

    std::vector<__uint128_t> inputs;

    // Structural edge cases
    inputs.push_back(0);
    inputs.push_back(~__uint128_t(0));          // all bits set
    inputs.push_back(1);                         // only bit 0
    inputs.push_back(__uint128_t(1) << 127);     // only bit 127
    inputs.push_back(__uint128_t(1) << 63);      // straddles the two 64-bit halves
    inputs.push_back(__uint128_t(0xAAAAAAAAAAAAAAAAULL) << 64 |
                     __uint128_t(0xAAAAAAAAAAAAAAAAULL)); // alternating

    // Reproducible random values
    std::mt19937_64 rng(0xDEADBEEF);
    for (int i = 0; i < 20; i++) {
        inputs.push_back((__uint128_t(rng()) << 64) | rng());
    }

    bool ok = true;
    for (auto v : inputs) {
        if (!check_one(p, v, perm, test_name.c_str())) {
            ok = false;
        }
    }

    std::cout << (ok ? "  PASSED" : "  FAILED") << "  " << test_name << "\n";
    return ok;
}

template<typename Perm128>
static int test_all(const std::string& impl_name)
{
    std::cout << "\n=== " << impl_name << " ===\n";

    int passed = 0, total = 0;
    std::vector<uint8_t> perm(128);

    auto run = [&](const std::string& name) {
        total++;
        if (test_one_perm<Perm128>(perm, name)) passed++;
    };

    // Identity
    for (int i = 0; i < 128; i++) perm[i] = i;
    run("identity");

    // Bit-reverse
    for (int i = 0; i < 128; i++) perm[i] = 127 - i;
    run("reverse");

    // Swap adjacent pairs (0↔1, 2↔3, …)
    for (int i = 0; i < 128; i++) perm[i] = (i & ~1) | (1 - (i & 1));
    run("swap_adjacent_pairs");

    // Rotate left by 1
    for (int i = 0; i < 128; i++) perm[i] = (i + 1) % 128;
    run("rotate_left_1");

    // Rotate left by 7 (crosses byte boundaries)
    for (int i = 0; i < 128; i++) perm[i] = (i + 7) % 128;
    run("rotate_left_7");

    // Rotate left by 64 (swaps upper/lower halves)
    for (int i = 0; i < 128; i++) perm[i] = (i + 64) % 128;
    run("rotate_left_64");

    // Swap upper/lower 64-bit words bit by bit
    for (int i = 0; i < 64; i++) { perm[i] = i + 64; perm[i + 64] = i; }
    run("swap_halves");

    // Five independent random permutations (different seeds)
    for (int seed = 0; seed < 5; seed++) {
        for (int i = 0; i < 128; i++) perm[i] = i;
        std::mt19937 rng(seed * 1234567 + 42);
        std::shuffle(perm.begin(), perm.end(), rng);
        run("random_" + std::to_string(seed));
    }

    std::cout << passed << "/" << total << " passed\n";
    return (passed == total) ? 0 : 1;
}


int main()
{
    int rc = 0;

#if defined(__AVX512F__)
    rc |= test_all<Permute128_AVX512>("AVX512");
#endif
#if defined(__AVX2__)
    rc |= test_all<Permute128_AVX2>("AVX2");
#endif
    rc |= test_all<Permute128_fallback>("fallback");

    return rc;
}
