#include "operator.hpp"
#include <cassert>
#include <cstdlib>

void ZBasisBase::search_batch(const state_t* q, idx_t n, idx_t* out) const {
    const __uint128_t* arr = reinterpret_cast<const __uint128_t*>(states.data());
    const idx_t N = (idx_t)states.size();
    if (N == 0) { for (idx_t i = 0; i < n; ++i) out[i] = -1; return; }

    // GROUP independent binary searches advance in lockstep: issue all their
    // probe prefetches, THEN do all their comparisons, so each probe's miss has
    // ~GROUP loads' worth of time to resolve before it is used. GROUP is the
    // number of memory requests kept in flight; tune via APPLY_SEARCH_GROUP
    // (read once, clamped to [1, MAX_GROUP]).
    constexpr int MAX_GROUP = 32;
    static const int GROUP = [] {
        const char* e = std::getenv("APPLY_SEARCH_GROUP");
        int g = e ? std::atoi(e) : 8;
        return g < 1 ? 1 : (g > MAX_GROUP ? MAX_GROUP : g);
    }();

    int iters = 1;
    while ((idx_t(1) << iters) < N) ++iters;
    ++iters;                              // guarantee lo==hi convergence

    idx_t lo[MAX_GROUP], hi[MAX_GROUP], mid[MAX_GROUP];
    for (idx_t base = 0; base < n; base += GROUP) {
        const int g = (int)std::min<idx_t>(GROUP, n - base);
        for (int k = 0; k < g; ++k) { lo[k] = 0; hi[k] = N; }

        for (int it = 0; it < iters; ++it) {
            for (int k = 0; k < g; ++k) {
                mid[k] = (lo[k] + hi[k]) >> 1;
                __builtin_prefetch(arr + mid[k], 0, 0);
            }
            for (int k = 0; k < g; ++k) {
                if (lo[k] >= hi[k]) continue;
                if (arr[mid[k]] < q[base + k].uint128) lo[k] = mid[k] + 1;
                else                                   hi[k] = mid[k];
            }
        }
        for (int k = 0; k < g; ++k) {
            const idx_t p = lo[k];
            out[base + k] = (p < N && arr[p] == q[base + k].uint128) ? p : -1;
        }
    }
}


int ZBasisBST::search(const state_t& state, idx_t& J) const {
    const __uint128_t* arr = reinterpret_cast<const __uint128_t*>(states.data());
    int64_t left = 0, right = states.size() - 1;

    static const int64_t CACHE_SIZE=32;
    while (right - left > CACHE_SIZE) {
        size_t mid = (left + right) / 2;
        
        if (arr[mid] < state.uint128) left = mid + 1;
        else right = mid;
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
    const __uint128_t* arr = reinterpret_cast<const __uint128_t*>(states.data());
    uint64_t key = state.uint64[1] & hi_mask;
    auto it = bounds.find(key);
    if (it == bounds.end()) return 0;
    auto [left, right] = it->second;

    static const idx_t CACHE_SIZE=32;

    // When hi_mask < ~0ULL the range may span multiple distinct uint64[1] values,
    // so compare the full 128-bit value rather than just uint64[0].
    while (right - left > CACHE_SIZE) {
        idx_t mid = (left + right) / 2;
        if (arr[mid] < state.uint128)
            left = mid + 1;
        else
            right = mid;
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
    for (idx_t J = 0; J < dim(); J++) {
        uint64_t key = states[J].uint64[1] & hi_mask;
        auto it = bounds.find(key);
        if (it != bounds.end()) {
            it->second.second = J;
        } else {
            bounds[key].first  = J;
            bounds[key].second = J;
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
    on_states_changed(); // rebuild bounds table, sentinel index, etc.
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


