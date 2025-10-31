#pragma once
#include <cstddef>
#include <array>
#include <vector>
#include <bit>
#include "old_gcc_shims.hpp"

namespace Benes {

template <typename UintT>
class BenesNetwork {
private:
    static constexpr size_t N = sizeof(UintT) * 8;
    static constexpr size_t STAGES = 2 * std::bit_width(N - 1) - 1;
    
    // Switch settings for each stage (bit i indicates whether position i swaps with i+delta)
    std::array<UintT, STAGES> switches;
    
    // Recursive routing for Benes network
    void route_recursive(const std::vector<size_t>& perm, size_t stage, size_t offset, size_t size) {
        if (size <= 1) return;
        
        size_t half = size / 2;
        std::vector<bool> upper_switches(half, false);
        std::vector<bool> used(size, false);
        std::vector<size_t> upper_perm(half), lower_perm(half);
        
        // Route using the looping algorithm
        for (size_t start = 0; start < size; start++) {
            if (used[start]) continue;
            
            size_t pos = start;
            bool in_upper = (pos < half);
            
            while (!used[pos]) {
                used[pos] = true;
                size_t dest = perm[pos];
                bool dest_upper = (dest < half);
                
                if (in_upper) {
                    upper_switches[pos] = (dest_upper != in_upper);
                } else {
                    upper_switches[pos - half] = (dest_upper != in_upper);
                }
                
                // Move to paired position
                pos = (pos < half) ? (pos + half) : (pos - half);
                in_upper = !in_upper;
                
                if (used[pos]) break;
                used[pos] = true;
                
                // Follow the permutation
                dest = perm[pos];
                pos = dest;
                in_upper = (pos < half);
            }
        }
        
        // Set switches for current stage
        for (size_t i = 0; i < half; i++) {
            if (upper_switches[i]) {
                switches[stage] |= (UintT(1) << (offset + i));
            }
        }
        
        // Build sub-permutations
        std::vector<size_t> upper_idx, lower_idx;
        for (size_t i = 0; i < size; i++) {
            size_t dest = perm[i];
            bool swap = (i < half) ? upper_switches[i] : upper_switches[i - half];
            
            if (i < half) {
                if (!swap) {
                    upper_idx.push_back(dest < half ? dest : dest - half);
                } else {
                    lower_idx.push_back(dest < half ? dest : dest - half);
                }
            } else {
                if (!swap) {
                    lower_idx.push_back(dest < half ? dest : dest - half);
                } else {
                    upper_idx.push_back(dest < half ? dest : dest - half);
                }
            }
        }
        
        for (size_t i = 0; i < half; i++) {
            upper_perm[i] = upper_idx[i];
            lower_perm[i] = lower_idx[i];
        }
        
        // Recursively route sub-networks
        if (half > 1) {
            route_recursive(upper_perm, stage + 1, offset, half);
            route_recursive(lower_perm, stage + 1, offset + half, half);
        }
        
        // Set switches for final stage (mirror of first)
        size_t mirror_stage = STAGES - 1 - stage;
        for (size_t i = 0; i < half; i++) {
            if (upper_switches[i]) {
                switches[mirror_stage] |= (UintT(1) << (offset + i));
            }
        }
    }
    
public:
    template <typename T>
    requires std::convertible_to<T, size_t>
    BenesNetwork(const std::vector<T>& I) {
        switches.fill(0);
        
        // Convert to size_t permutation
        std::vector<size_t> perm(N);
        for (size_t i = 0; i < N; i++) {
            perm[i] = (i < I.size()) ? static_cast<size_t>(I[i]) : i;
        }
        
        route_recursive(perm, 0, 0, N);
    }
    
    UintT apply(UintT x) const {
        // Apply Benes network stages
        for (size_t stage = 0; stage < STAGES; stage++) {
            size_t depth = (stage < STAGES / 2) ? stage : (STAGES - 1 - stage);
            size_t delta = 1 << (std::bit_width(N - 1) - 1 - depth);
            UintT mask = switches[stage];
            
            // For each switch position, conditionally swap with position + delta
            UintT bits_to_swap = mask & (x ^ (x >> delta));
            x ^= bits_to_swap | (bits_to_swap << delta);
        }
        return x;
    }
};

template <typename UintT, typename T>
requires std::convertible_to<T, size_t>
inline UintT permute(const UintT& x, const std::vector<T>& I) {
    BenesNetwork<UintT> network(I);
    return network.apply(x);
}

};
