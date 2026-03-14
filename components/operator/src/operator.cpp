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

//    for (J = left; J <= right; J++) {
//        if (arr[J] == state.uint128) return 1;
//    }

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

//    for (J = left; J <= right; J++) {
//        if (arr[J] == state.uint128) return 1;
//    }

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
    for (idx_t J = 0; J < dim(); J++) {
        uint64_t state_hi = states[J].uint64[1];
        auto it = bounds.find(state_hi);
        if (it != bounds.end()) {
            it->second.second = J;
        } else {
            bounds[state_hi].first  = J;
            bounds[state_hi].second = J;
        }
    }
    
}


size_t insert_states(std::vector<ZBasisBST::state_t>& states,
                     std::vector<ZBasisBST::state_t>& to_insert) {
    size_t n_insertions = 0;

    std::sort(to_insert.begin(), to_insert.end());
    to_insert.erase(std::unique(to_insert.begin(), to_insert.end()), to_insert.end());

    std::vector<ZBasisBST::state_t> merged;
    merged.reserve(states.size() + to_insert.size());

    auto it_old = states.begin();
    auto it_new = to_insert.begin();

    while (it_old != states.end() && it_new != to_insert.end()) {
        if (*it_new < *it_old) {
            merged.push_back(*it_new);
            ++n_insertions;
            ++it_new;
        } else if (*it_old < *it_new) {
            merged.push_back(*it_old);
            ++it_old;
        } else {
            merged.push_back(*it_old);
            ++it_old;
            ++it_new;
        }
    }

    while (it_new != to_insert.end()) {
        merged.push_back(*it_new);
        ++n_insertions;
        ++it_new;
    }
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


void ZBasisBSTFast::build_sentinels() {
    // Target ~4 MB for the sentinel array so it reliably fits in L3 cache.
    // For N ~ 1e9 this gives stride ~ 4096, reducing cold DRAM accesses during
    // search from log2(N) ~ 30 to log2(stride) ~ 12.
    static constexpr size_t TARGET_BYTES = 4ULL * 1024 * 1024;
    const idx_t n = dim();
    const idx_t max_sentinels = static_cast<idx_t>(TARGET_BYTES / sizeof(state_t));
    stride = std::max<idx_t>(1, n / max_sentinels);

    sentinels.clear();
    sentinels.reserve((n + stride - 1) / stride);
    for (idx_t i = 0; i < n; i += stride)
        sentinels.push_back(states[i]);
}

void ZBasisBSTFast::load_from_file(const fs::path& bfile, const std::string& dataset){
    this->ZBasisBase::load_from_file(bfile, dataset);
    build_sentinels();
}

int ZBasisBSTFast::search(const state_t& state, idx_t& J) const {
    const __uint128_t* arr  = reinterpret_cast<const __uint128_t*>(states.data());
    const __uint128_t* sarr = reinterpret_cast<const __uint128_t*>(sentinels.data());

    // Step 1: binary-search the warm sentinel index.
    // Finds sl = first sentinel index where sentinels[sl] >= state.
    // After this, the target (if present) lies in states[lo .. hi]
    // where hi - lo <= stride, costing only log2(stride) cold DRAM accesses.
    idx_t sl = 0, sr = (idx_t)sentinels.size() - 1;
    while (sl < sr) {
        idx_t sm = sl + (sr - sl) / 2;
        if (sarr[sm] < state.uint128) sl = sm + 1;
        else sr = sm;
    }

    // Derive [lo, hi] from sentinel position.
    // sentinels[sl] = states[sl*stride] is the first sentinel >= state,
    // so the target lies between states[(sl-1)*stride] and states[sl*stride].
    idx_t lo = (sl > 0) ? (sl - 1) * stride : 0;
    idx_t hi = (sl < (idx_t)sentinels.size() - 1) ? sl * stride : dim() - 1;

    // Step 2: binary search within [lo, hi] (at most stride+1 elements).
    static const idx_t CACHE_SIZE = 32;
    while (hi - lo > CACHE_SIZE) {
        idx_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < state.uint128) lo = mid + 1;
        else hi = mid;
    }

    for (J = lo; J + 3 <= hi; J += 4) {
        if (arr[J]   == state) {             return 1; }
        if (arr[J+1] == state) { J = J + 1; return 1; }
        if (arr[J+2] == state) { J = J + 2; return 1; }
        if (arr[J+3] == state) { J = J + 3; return 1; }
    }
    for (; J <= hi; ++J) {
        if (arr[J] == state) return 1;
    }
    return 0;
}


