#include <cassert>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include "bittools.hpp" // Replace with your actual header filename

void test_constructors() {
    std::cout << "Testing constructors..." << std::endl;
    
    // Default constructor
    Uint128 u1;
    
    // Single value constructor
    Uint128 u2(42);
    assert(u2.uint64[0] == 42);
    assert(u2.uint64[1] == 0);
    
    // Two uint64_t constructor
    Uint128 u3(0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL);
    assert(u3.uint64[0] == 0xFEDCBA9876543210ULL);
    assert(u3.uint64[1] == 0x123456789ABCDEF0ULL);
    
    // Template constructor with different types
    Uint128 u4(static_cast<uint32_t>(100));
    assert(u4.uint64[0] == 100);
    assert(u4.uint64[1] == 0);
}

void test_operators() {
    std::cout << "Testing operators..." << std::endl;
    
    Uint128 a(0x1111111111111111ULL, 0x2222222222222222ULL);
    Uint128 b(0x3333333333333333ULL, 0x4444444444444444ULL);
    
    // Test XOR assignment
    Uint128 c = a;
    c ^= b;
    assert(c.uint64[0] == (0x2222222222222222ULL ^ 0x4444444444444444ULL));
    assert(c.uint64[1] == (0x1111111111111111ULL ^ 0x3333333333333333ULL));
    
    // Test less than operator
    Uint128 small(1, 0);
    Uint128 large(2, 0);
    assert(small < large);
    
    Uint128 small_high(0, 1);
    Uint128 large_high(0, 2);
    assert(small_high < large_high);
    
    // Test equality operator
    Uint128 equal1(123, 456);
    Uint128 equal2(123, 456);
    assert(equal1 == equal2);
    
    // Test AND operator
    Uint128 and_result = a & b;
    assert(and_result.uint64[0] == (0x2222222222222222ULL & 0x4444444444444444ULL));
    assert(and_result.uint64[1] == (0x1111111111111111ULL & 0x3333333333333333ULL));
    
    // Test XOR operator
    Uint128 xor_result = a ^ b;
    assert(xor_result.uint64[0] == (0x2222222222222222ULL ^ 0x4444444444444444ULL));
    assert(xor_result.uint64[1] == (0x1111111111111111ULL ^ 0x3333333333333333ULL));
    
    // Test OR operator
    Uint128 or_result = a | b;
    assert(or_result.uint64[0] == (0x2222222222222222ULL | 0x4444444444444444ULL));
    assert(or_result.uint64[1] == (0x1111111111111111ULL | 0x3333333333333333ULL));
}

void test_shift_operators() {
    std::cout << "Testing shift operators..." << std::endl;
    
    // Test left shift
    Uint128 val(0, 1);
    Uint128 shifted_left = val << 1;
    assert(shifted_left.uint64[0] == 2);
    assert(shifted_left.uint64[1] == 0);
    
    // Test right shift
    Uint128 val2(0, 4);
    Uint128 shifted_right = val2 >> 2;
    assert(shifted_right.uint64[0] == 1);
    assert(shifted_right.uint64[1] == 0);
    
    // Test large shift
    Uint128 val3(0, 1);
    Uint128 shifted_left_64 = val3 << 65;
    assert(shifted_left_64.uint64[0] == 0);
    assert(shifted_left_64.uint64[1] == 2);

    Uint128 val4(2, 0);
    Uint128 shifted_right_64 = val4 >> 65;
    assert(shifted_right_64.uint64[0] == 1);
    assert(shifted_right_64.uint64[1] == 0);
}

void test_bit_operations() {
    std::cout << "Testing bit operations..." << std::endl;
    
    Uint128 x(0, 0);
    
    // Test or_bit function
    or_bit(x, 0);
    assert(x.uint64[0] == 1);
    
    or_bit(x, 63);
    assert(x.uint64[0] == (1ULL | (1ULL << 63)));
    
    or_bit(x, 64);
    assert(x.uint64[1] == 1);
    
    or_bit(x, 127);
    assert(x.uint64[1] == (1ULL | (1ULL << 63)));
    
    // Test readbit function
    assert(readbit(x, 0) == true);
    assert(readbit(x, 1) == false);
    assert(readbit(x, 63) == true);
    assert(readbit(x, 64) == true);
    assert(readbit(x, 127) == true);
    assert(readbit(x, 65) == false);
}

void test_popcnt() {
    std::cout << "Testing population count..." << std::endl;
    
    // Test empty
    Uint128 empty(0, 0);
    assert(popcnt_u128(empty) == 0);
    
    // Test single bit
    Uint128 single(0, 1);
    assert(popcnt_u128(single) == 1);
    
    // Test all bits in low 64
    Uint128 all_low(0, 0xFFFFFFFFFFFFFFFFULL);
    assert(popcnt_u128(all_low) == 64);
    
    // Test all bits in high 64
    Uint128 all_high(0xFFFFFFFFFFFFFFFFULL, 0);
    assert(popcnt_u128(all_high) == 64);
    
    // Test mixed
    Uint128 mixed(0xF0F0F0F0F0F0F0F0ULL, 0x0F0F0F0F0F0F0F0FULL);
    assert(popcnt_u128(mixed) == 64);
    
    // Test specific pattern
    Uint128 pattern(0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL);
    assert(popcnt_u128(pattern) == 64);
}

void test_make_mask() {
    std::cout << "Testing make_mask..." << std::endl;
    
    // Test mask of 0 bits
    Uint128 mask0 = make_mask(0);
    assert(mask0.uint64[0] == 0);
    assert(mask0.uint64[1] == 0);
    
    // Test mask of 1 bit
    Uint128 mask1 = make_mask(1);
    assert(mask1.uint64[0] == 1);
    assert(mask1.uint64[1] == 0);
    
    // Test mask of 8 bits
    Uint128 mask8 = make_mask(8);
    assert(mask8.uint64[0] == 0xFF);
    assert(mask8.uint64[1] == 0);
    
    // Test mask of 64 bits
    Uint128 mask64 = make_mask(64);
    assert(mask64.uint64[0] == 0xFFFFFFFFFFFFFFFFULL);
    assert(mask64.uint64[1] == 0);
    
    // Test mask of 65 bits
    Uint128 mask65 = make_mask(65);
    assert(mask65.uint64[0] == 0xFFFFFFFFFFFFFFFFULL);
    assert(mask65.uint64[1] == 1);
    
    // Test mask of 128 bits (edge case)
    Uint128 mask128 = make_mask(127);
    assert(mask128.uint64[0] == 0xFFFFFFFFFFFFFFFFULL);
    assert(mask128.uint64[1] == 0x7FFFFFFFFFFFFFFFULL);
}

void test_hash_and_equality() {
    std::cout << "Testing hash and equality functors..." << std::endl;
    
    Uint128 a(0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL);
    Uint128 b(0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL);
    Uint128 c(0x123456789ABCDEF1ULL, 0xFEDCBA9876543210ULL);
    
    Uint128Hash hasher;
    Uint128Eq eq;
    
    // Test equality functor
    assert(eq(a, b) == true);
    assert(eq(a, c) == false);
    
    // Test hash consistency
    assert(hasher(a) == hasher(b));
    
    // Test in unordered containers
    std::unordered_set<Uint128, Uint128Hash, Uint128Eq> set;
    set.insert(a);
    assert(set.find(b) != set.end());
    assert(set.find(c) == set.end());
    
    std::unordered_map<Uint128, int, Uint128Hash, Uint128Eq> map;
    map[a] = 42;
    assert(map[b] == 42);
    assert(map.find(c) == map.end());
}

void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Test maximum value
    Uint128 max_val(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
    
    // Test shifts at boundaries
    Uint128 boundary_shift = max_val >> 127;
    assert(boundary_shift.uint64[0] == 1);
    assert(boundary_shift.uint64[1] == 0);
    
    // Test XOR with self (should be zero)
    Uint128 self_xor = max_val ^ max_val;
    assert(self_xor.uint64[0] == 0);
    assert(self_xor.uint64[1] == 0);
    
    // Test AND with zero
    Uint128 zero(0, 0);
    Uint128 and_zero = max_val & zero;
    assert(and_zero.uint64[0] == 0);
    assert(and_zero.uint64[1] == 0);
    
    // Test OR with zero
    Uint128 or_zero = max_val | zero;
    assert(or_zero == max_val);
}

int main(int argc, char* argv[]) {
    std::cout << "Running Uint128 tests..." << std::endl;
    
    // If no arguments, run all tests
    if (argc == 1) {
        test_constructors();
        test_operators();
        test_shift_operators();
        test_bit_operations();
        test_popcnt();
        test_make_mask();
        test_hash_and_equality();
        test_edge_cases();
        std::cout << "All tests passed!" << std::endl;
        return 0;
    }
    
    // Run specific tests based on command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--test-constructors") {
            test_constructors();
        }
        else if (arg == "--test-operators") {
            test_operators();
            test_shift_operators();
        }
        else if (arg == "--test-bit-ops") {
            test_bit_operations();
            test_popcnt();
            test_make_mask();
        }
        else if (arg == "--test-hash") {
            test_hash_and_equality();
        }
        else if (arg == "--test-edge-cases") {
            test_edge_cases();
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --test-constructors  Run constructor tests\n";
            std::cout << "  --test-operators     Run operator tests\n";
            std::cout << "  --test-bit-ops       Run bit operation tests\n";
            std::cout << "  --test-hash          Run hash and equality tests\n";
            std::cout << "  --test-edge-cases    Run edge case tests\n";
            std::cout << "  --help, -h           Show this help message\n";
            std::cout << "  (no args)            Run all tests\n";
            return 0;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Use --help for usage information." << std::endl;
            return 1;
        }
    }
    
    std::cout << "Selected tests passed!" << std::endl;
    return 0;
}
