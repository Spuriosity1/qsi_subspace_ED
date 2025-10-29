#include "group_theory.hpp"
#include <queue>

template <typename T>

PermutationGroup<T>::PermutationGroup(std::vector<perm_t>& generators){
    // generates the full group and initialises the permuters
    if (generators.empty()) return;

    size_t n = generators[0].size();
    if (n > 128) {
        throw std::runtime_error("More than 128 sites needed, edit all the files");
    }

    std::set<perm_t> seen;
    std::queue<perm_t> todo;

    perm_t identity(n); 
    for (size_t i=0;i<n; i++) identity[i] = i;
    seen.insert(identity);
    todo.push(identity);

    for (const auto& g : generators){
        if (seen.insert(g).second) {
            todo.push(g);
        }
    }

    while (!todo.empty()){
        perm_t current = todo.front();
        todo.pop();
        // Multiply current by each generator
        for (const auto& gen : generators) {
            perm_t product = compose(current, gen);
            if (seen.insert(product).second) {
                todo.push(product);
            }
            // Also try generator * current
            product = compose(gen, current);
            if (seen.insert(product).second) {
                todo.push(product);
            }
        }
    }
    permuters.reserve(seen.size());
    group_elements.reserve(seen.size());
    for (const auto& perm : seen) {
        permuters.emplace_back(perm);
        group_elements.push_back(perm);
    }
}



// specialisations
template <>
PermutationGroup<Uint128>::PermutationGroup(std::vector<perm_t>& generators);
template <>
PermutationGroup<__int128>::PermutationGroup(std::vector<perm_t>& generators);
template <>
PermutationGroup<uint64_t>::PermutationGroup(std::vector<perm_t>& generators);
template <>
PermutationGroup<uint32_t>::PermutationGroup(std::vector<perm_t>& generators);
template <>
PermutationGroup<uint16_t>::PermutationGroup(std::vector<perm_t>& generators);
template <>
PermutationGroup<uint8_t>::PermutationGroup(std::vector<perm_t>& generators);

// signed (untested, use with great care)
template <>
PermutationGroup<int64_t>::PermutationGroup(std::vector<perm_t>& generators);
template <>
PermutationGroup<int32_t>::PermutationGroup(std::vector<perm_t>& generators);
template <>
PermutationGroup<int16_t>::PermutationGroup(std::vector<perm_t>& generators);
template <>
PermutationGroup<int8_t>::PermutationGroup(std::vector<perm_t>& generators);
