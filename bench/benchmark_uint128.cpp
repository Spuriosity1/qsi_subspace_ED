#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include "bittools.hpp" // Replace with your actual header filename

template<typename Func>
double benchmark_operation(const std::string& name, Func&& func, int iterations = 1000000) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double time_per_op = static_cast<double>(duration.count()) / iterations;
    std::cout << name << ": " << time_per_op << " μs per operation" << std::endl;
    
    return time_per_op;
}

int main() {
    std::cout << "Uint128 Performance Benchmarks" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // Generate test data
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    
    const int num_values = 1000;
    std::vector<Uint128> values;
    values.reserve(num_values);
    
    for (int i = 0; i < num_values; ++i) {
        values.emplace_back(dis(gen), dis(gen));
    }
    
    int counter = 0;
    
    // Benchmark construction
    benchmark_operation("Construction (two uint64_t)", [&]() {
        Uint128 temp(dis(gen), dis(gen));
        counter += temp.uint64[0] & 1; // Prevent optimization
    });
    
    // Benchmark XOR operation
    benchmark_operation("XOR operation", [&]() {
        auto& a = values[counter % num_values];
        auto& b = values[(counter + 1) % num_values];
        Uint128 result = a ^ b;
        counter += result.uint64[0] & 1;
    });
    
    // Benchmark AND operation
    benchmark_operation("AND operation", [&]() {
        auto& a = values[counter % num_values];
        auto& b = values[(counter + 1) % num_values];
        Uint128 result = a & b;
        counter += result.uint64[0] & 1;
    });
    
    // Benchmark OR operation
    benchmark_operation("OR operation", [&]() {
        auto& a = values[counter % num_values];
        auto& b = values[(counter + 1) % num_values];
        Uint128 result = a | b;
        counter += result.uint64[0] & 1;
    });
    
    // Benchmark left shift
    benchmark_operation("Left shift", [&]() {
        auto& a = values[counter % num_values];
        Uint128 result = a << (counter % 64);
        counter += result.uint64[0] & 1;
    });
    
    // Benchmark right shift
    benchmark_operation("Right shift", [&]() {
        auto& a = values[counter % num_values];
        Uint128 result = a >> (counter % 64);
        counter += result.uint64[0] & 1;
    });
    
    // Benchmark population count
    benchmark_operation("Population count", [&]() {
        auto& a = values[counter % num_values];
        int result = popcnt_u128(a);
        counter += result & 1;
    });
    
    // Benchmark bit operations
    benchmark_operation("Set bit (or_bit)", [&]() {
        Uint128 temp = values[counter % num_values];
        or_bit(temp, counter % 128);
        counter += temp.uint64[0] & 1;
    });
    
    benchmark_operation("Read bit", [&]() {
        auto& a = values[counter % num_values];
        bool result = readbit(a, counter % 128);
        counter += result ? 1 : 0;
    });
    
    // Benchmark comparison
    benchmark_operation("Less than comparison", [&]() {
        auto& a = values[counter % num_values];
        auto& b = values[(counter + 1) % num_values];
        bool result = a < b;
        counter += result ? 1 : 0;
    });
    
    benchmark_operation("Equality comparison", [&]() {
        auto& a = values[counter % num_values];
        auto& b = values[(counter + 1) % num_values];
        bool result = a == b;
        counter += result ? 1 : 0;
    });
    
    // Benchmark hash function
    Uint128Hash hasher;
    benchmark_operation("Hash function", [&]() {
        auto& a = values[counter % num_values];
        std::size_t result = hasher(a);
        counter += result & 1;
    });
    
    // Benchmark make_mask
    benchmark_operation("Make mask", [&]() {
        Uint128 result = make_mask(counter % 128);
        counter += result.uint64[0] & 1;
    });
    
    std::cout << "\nBenchmark completed. (counter=" << counter << ")" << std::endl;
    return 0;
}
