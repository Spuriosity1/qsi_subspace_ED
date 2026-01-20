#include "bittools.hpp"
#include "permute.hpp"
#include <algorithm>
#include <cassert>
#include <random>
#include <cstdio>
#include <chrono>

using UintT = Uint128;


volatile __uint128_t sink;

template<typename Perm>
auto bench(const std::string& name, const std::vector<Uint128>& test_values){
    std::vector<uint8_t> perm(128);
    for(int i=0; i<128; i++)
        perm[i]=i;
    std::mt19937 rng;
    std::shuffle(perm.begin(), perm.end(), rng);

    Perm p(perm);


    std::cout << "Benchmarking "<<name<<" implementation\n";
    auto start = std::chrono::steady_clock::now();
    for (auto psi : test_values){
        sink = p.permute(psi.uint128);
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed_seconds = finish - start;

    auto nreps = test_values.size();

    std::cout << "Duration: "<<elapsed_seconds.count()/nreps<<" per call\n";
    return elapsed_seconds;
}


int main(int argc, char* argv[]) {

    std::cout << "=== Permutation Benchmark ===" << std::endl << std::endl;
    std::vector<uint8_t> perm(128);
    for(int i=0; i<128; i++)
        perm[i]=i;
    std::mt19937 rng;
    std::shuffle(perm.begin(), perm.end(), rng);

    if (argc < 2){
        printf("usage: %s NREPS\n", argv[0]);
        return 1;
    }
    int nreps = atoi(argv[1]);


    std::vector<UintT> test_values;

    // Random values
    for (int i = 0; i < nreps; i++) {
        UintT val;
        if constexpr (sizeof(UintT) <= 8) {
            val = rng();
        } else {
            // For Uint128, combine two random 64-bit values
            val = (UintT(rng()) << 64) | UintT(rng());
        }
        test_values.push_back(val);
    }


    std::cout << "Benchmarking NAIVE implementation\n";
    auto start{std::chrono::steady_clock::now()};
    for (auto psi : test_values){
        sink = permute(psi.uint128, perm);
    }
    auto finish{std::chrono::steady_clock::now()};
    std::chrono::duration<double> elapsed_seconds{finish - start};
    std::cout << "Duration: "<< elapsed_seconds.count()/nreps<<"\n";
    
    bench<Permute128_fallback>("DEFAULT", test_values);

#if defined(__AVX512F__)
    bench<Permute128_AVX512>("AVX512", test_values);
#endif
#if defined(__AVX2__)
    bench<Permute128_AVX2>("AVX2", test_values);
#endif


}

