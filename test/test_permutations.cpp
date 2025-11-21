#include "bittools.hpp"
#include "permute.hpp"
#include <algorithm>
#include <cassert>
#include <random>

using UintT = Uint128;

// Test function
template<typename Perm128>
bool test_benes_permutation(const std::vector<uint8_t>& perm, const std::string& test_name) {
    constexpr size_t N = sizeof(UintT) * 8;
    assert(perm.size() == N);
    
    std::cout << "Running test: " << test_name << std::endl;
    
    // Build Benes network
    Perm128 p(perm);
    
    // Test multiple input values
    std::vector<UintT> test_values;
    
    // Edge cases
    test_values.push_back(0);
    test_values.push_back(~UintT(0)); // All bits set
    test_values.push_back(1);
    test_values.push_back(UintT(1) << (N - 1)); // MSB set
    
    // Random values
    std::mt19937_64 rng(42);
    for (int i = 0; i < 10; i++) {
        UintT val;
        if constexpr (sizeof(UintT) <= 8) {
            val = rng();
        } else {
            // For Uint128, combine two random 64-bit values
            val = (UintT(rng()) << 64) | UintT(rng());
        }
        test_values.push_back(val);
    }

    
    // Test each value
    bool all_passed = true;
    for (size_t i = 0; i < test_values.size(); i++) {
        std::cout << i<<"]\r"<<std::flush;
        UintT input = test_values[i];
        UintT benes_result = p.permute(input.uint128);
        UintT naive_result = permute(input, perm);
        
        if (benes_result != naive_result) {
            std::cout << "  FAILED on test value " << i << std::endl;
            std::cout << "    Input: " <<  input << std::endl;
            std::cout << "    Perm: " <<  benes_result << std::endl;
            std::cout << "    Naive: " <<  naive_result << std::endl;
            
            all_passed = false;
        }
    }
    
    if (all_passed) {
        std::cout << "  PASSED ✓" << std::endl;
    }
    std::cout << std::endl;
    
    return all_passed;
}

template<typename Perm>
int test_all(const std::string& name){

    std::cout << "=== "<<name<<" Permutation Tests ===" << std::endl << std::endl;
    
    int passed = 0;
    int total = 0;

    std::vector<uint8_t> perm(128);
    
    // Test 1: Identity permutation (8-bit)
    {
        for(int i=0; i<128; i++)
            perm[i]=i;

        total++;
        if (test_benes_permutation<Perm>(perm, "Identity ")) passed++;
    }
    
    // Test 2: Reverse permutation 
    {
        for(int i=0; i<128; i++)
            perm[i]=127-i;
        total++;
        if (test_benes_permutation<Perm>(perm, "Reverse ")) passed++;
    }
    
    // Test 3: Swap adjacent pairs 
    {
        for(int i=0; i<128; i++)
            perm[i]= 2*(i/2) + 1 - i%2;
        total++;
        if (test_benes_permutation<Perm>(perm, "Swap pairs ")) passed++;
    }
    
    // Test 4: Rotate left 
    {
        for(int i=0; i<128; i++)
            perm[i]=(i+1) % 128;
        total++;
        if (test_benes_permutation<Perm>(perm, "Rotate left ")) passed++;
    }
    
    // Test 5: Random permutation 
    {
        std::mt19937 rng;
        for(int i=0; i<128; i++)
            perm[i]=i;
        std::shuffle(perm.begin(), perm.end(), rng);
        total++;
        if (test_benes_permutation<Perm>(perm, "Random ")) passed++;
    }
      
    std::cout << "==================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    
    if (passed == total) {
        std::cout << "All tests PASSED! ✓" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED! ✗" << std::endl;
        return 1;
    }
}


int main() {
  int errors=0;
  int attempted = 0;
#if defined(__AVX512F__)
errors += test_all<Permute128_AVX512>("AVX512");
attempted++;
#endif
#if defined(__AVX2__)
errors += test_all<Permute128_AVX2>("AVX2");
attempted++;
#endif
errors += test_all<Permute128_fallback>("default");
attempted++;

std::cout << "============================\n"
          << "==         Summary        ==\n"
          << "============================\n"
          << (attempted - errors) << " / " << attempted << " tests succeeeded" <<
          std::endl;


}
