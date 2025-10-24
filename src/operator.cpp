#include "operator.hpp"
#include <cassert>

int ZBasisBST::search(const state_t& state, idx_t& J) const {
//    auto it = std::lower_bound(states.begin(), states.end(), state);
//    return it;
    
    const __uint128_t* arr = reinterpret_cast<const __uint128_t*>(states.data());
    int64_t left = 0, right = states.size() - 1;

    static const int64_t CACHE_SIZE=32;
    
    while (right - left > CACHE_SIZE) {
        size_t mid = (left + right) / 2;
        
        if (arr[mid] < state.uint128) left = mid + 1;
        else right = mid;
    }

    for (J = left; J <= right; J++) {
        if (arr[J] == state.uint128) return 1;
    }

    // manual unroll BS (actually saves noticeable time???)
    for (J = left; J + 3 <= right; J += 4) {
        if (arr[J] == state) {  return 1; }
        if (arr[J+1] == state) { J = J+1; return 1; }
        if (arr[J+2] == state) { J = J+2; return 1; }
        if (arr[J+3] == state) { J = J+3; return 1; }
    }
    for (; J <= right; ++J) {
        if ( J >= 0 && arr[J] == state) { return 1; }
    }
    return 0; // not found;
}



int ZBasisInterp::search(const state_t& state, idx_t& J) const {
//    auto it = std::lower_bound(states.begin(), states.end(), state);
//    return it;
//
    const __uint128_t* arr = reinterpret_cast<const __uint128_t*>(states.data());
    auto [left, right] = bounds.at(state.uint64[1]);

    static const idx_t CACHE_SIZE=32;
    
    while (right - left > CACHE_SIZE) {
        idx_t mid = (left + right) / 2;
        
        if (*reinterpret_cast<const uint64_t*>(arr + mid) < state.uint64[0]) 
            left = mid + 1;
        else 
            right = mid;
    }

    for (J = left; J <= right; J++) {
        if (arr[J] == state.uint128) return 1;
    }

    for (J = left; J + 3 <= right; J += 4) {
        if (arr[J] == state) {  return 1; }
        if (arr[J+1] == state) { J = J+1; return 1; }
        if (arr[J+2] == state) { J = J+2; return 1; }
        if (arr[J+3] == state) { J = J+3; return 1; }
    }
    for (; J <= right; ++J) {
        if (J>=0 && arr[J] == state) { return 1; }
    }
    return 0; // not found;
}


void ZBasisInterp::find_bounds(){
    bounds.clear();
    for (idx_t J=0; J<dim(); J++){
        uint64_t state_hi = states[J].uint64[1];
        if (bounds.contains(state_hi)){
            bounds[state_hi].second = J;
        } else {
            bounds[state_hi].first = J;
            bounds[state_hi].second = J;
        }
    }
    
}


// Inserts "to_insert" into the basis, storing the new states in new_states.
// Ensures they are inserted in sorted order.
size_t ZBasisBST::insert_states(std::vector<ZBasisBST::state_t>& to_insert){
    size_t n_insertions = 0;

    // sort to_insert and remove duplicates
    std::sort(to_insert.begin(), to_insert.end());
    to_insert.erase(std::unique(to_insert.begin(), to_insert.end()), to_insert.end());

    std::vector<ZBasisBST::state_t> merged;
    merged.reserve(states.size() + to_insert.size());
    
    auto it_old = states.begin();
    auto it_new = to_insert.begin();

    while (it_old != states.end() && it_new != to_insert.end()){
        if (*it_new < *it_old) {
            // New unique state to insert
            merged.push_back(*it_new);
            ++n_insertions;
            ++it_new;
        } else if (*it_old < *it_new) {
            merged.push_back(*it_old);
            ++it_old;
        } else {
            // Duplicate; keep existing
            merged.push_back(*it_old);
            ++it_old;
            ++it_new;
        }
    }

    // Append any remaining new states
    while (it_new != to_insert.end()) {
        merged.push_back(*it_new);
        ++n_insertions;
        ++it_new;
    }

    // Append remaining old states
    while (it_old != states.end()) {
        merged.push_back(*it_old);
        ++it_old;
    }
    
    states.swap(merged);
    return n_insertions;
}

void ZBasisBase::load_from_file(const fs::path& bfile, const std::string& dataset){
    std::cerr << "Loading basis from file " << bfile <<"\n";
    if (bfile.stem().extension() == ".partitioned"){
        assert(bfile.extension() == ".h5");
        states = basis_io::read_basis_hdf5(bfile, dataset.c_str());
    } else if (bfile.extension() == ".h5"){
        assert(dataset=="basis");
        states = basis_io::read_basis_hdf5(bfile); 
    } else if (bfile.extension() == ".csv"){
        assert(dataset=="basis");
        states = basis_io::read_basis_csv(bfile); 
    } else {
        throw std::runtime_error(
                "Bad basis format: file must end with .csv or .h5");
    }
}


void ZBasisInterp::load_from_file(const fs::path& bfile, const std::string& dataset){
    this->ZBasisBase::load_from_file(bfile, dataset);
    find_bounds();
}


