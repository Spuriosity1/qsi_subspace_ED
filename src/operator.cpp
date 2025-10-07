#include "operator.hpp"
#include <omp.h>

bool ZBasisBST::search(const state_t& state, idx_t& J) const {
//    auto it = std::lower_bound(states.begin(), states.end(), state);
//    return it;
    const __uint128_t* arr = reinterpret_cast<const __uint128_t*>(states.data());
    size_t left = 0, right = states.size() - 1;

    static const size_t CACHE_SIZE=32;
    
    while (right - left > CACHE_SIZE) {
        size_t mid = (left + right) / 2;
        
        if (arr[mid] < state.uint128) left = mid + 1;
        else right = mid;
    }

    for (J = left; J <= right; J++) {
        if (arr[J] == state.uint128) return true;
    }

    // manual unroll BS (actually saves noticeable time???)
    for (J = left; J + 3 <= right; J += 4) {
        if (arr[J] == state) {  return true; }
        if (arr[J+1] == state) { J = J+1; return true; }
        if (arr[J+2] == state) { J = J+2; return true; }
        if (arr[J+3] == state) { J = J+3; return true; }
    }
    for (; J <= right; ++J) {
        if (arr[J] == state) { J = J; return true; }
    }
    return false; // not found;
}



bool ZBasisInterp::search(const state_t& state, idx_t& J) const {
//    auto it = std::lower_bound(states.begin(), states.end(), state);
//    return it;
//
    const __uint128_t* arr = reinterpret_cast<const __uint128_t*>(states.data());
    auto [left, right] = bounds.at(state.uint64[1]);

    static const size_t CACHE_SIZE=32;
    
    while (right - left > CACHE_SIZE) {
        size_t mid = (left + right) / 2;
        
        if (*reinterpret_cast<const uint64_t*>(arr + mid) < state.uint64[0]) 
            left = mid + 1;
        else 
            right = mid;
    }

    for (J = left; J <= right; J++) {
        if (arr[J] == state.uint128) return true;
    }

    for (J = left; J + 3 <= right; J += 4) {
        if (arr[J] == state) { J = J; return true; }
        if (arr[J+1] == state) { J = J+1; return true; }
        if (arr[J+2] == state) { J = J+2; return true; }
        if (arr[J+3] == state) { J = J+3; return true; }
    }
    for (; J <= right; ++J) {
        if (arr[J] == state) { J = J; return true; }
    }
    return false; // not found;
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



size_t ZBasisBST::insert_states(const std::vector<ZBasisBST::state_t>& to_insert,
        std::vector<ZBasisBST::state_t>& new_states){
    new_states.resize(0);
    size_t n_insertions = 0;
    for (auto& s : to_insert){
        ZBasisBST::idx_t tmp;
        // skip if we know about it already
        if (this->search(s, tmp)) continue;
//        state_to_index[s] = states.size();
        states.push_back(s);
        new_states.push_back(s);
        n_insertions++;
    }
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


