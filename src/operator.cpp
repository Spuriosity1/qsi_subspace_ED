#include "operator.hpp"
#include <omp.h>
//
//void UInt128map::initialise(const std::vector<state_t>& states, int n_spins,
//        int n_radix) {
//    // do-once fn. can be as slow as you like
//    // well not really. Insist that the radix fits wholly within either 'hi' or 'lo'
//    // for better performance
//
//    // n.b. assumes psi >> n_spins = 0 
//    this->n_spins = n_spins;
//    this->n_radix = n_radix;
//    
//    if ( n_spins <= 64 ){
//        if ( n_radix > n_spins ) {
//            throw std::logic_error("Radix cannot be greater than # spins");
//        }
//        initialise_lt64(states);
//    } else {
//        if ( n_radix > n_spins - 64 ) {
//            throw std::logic_error("Radix cannot be greater than # spins - 64");
//        }
//
//        initialise_gt64(states);
//    }
//}
//
//void UInt128map::initialise_lt64(const std::vector<state_t>& states){
//    for (idx_t i=0; i<states.size(); i++){
//        
//    }
//}
//
//
//void UInt128map::initialise_gt64(const std::vector<state_t>& states){
//    bounds.reserve((1 << n_radix) + 1);
//    bounds.resize(0);
//    for (int super=0; super <= (1<<n_radix); super++){
//        bounds[super] 
//    }
//
//    hi_shift = (n_spins - 64 - n_radix);
//    hi_mask = ~(0ull) >> (64 - n_radix) << hi_shift;
//    // masks off only the radix bits
//
//    for (idx_t i=0; i<states.size(); i++){
//        uint64_t super = (states[i].uint64[1] & hi_mask) >> hi_shift;
//
//    }
//}

bool ZBasisHashmap::search(const state_t& state, idx_t& J) const {
//    J = phash(state);
    auto it = state_to_index.find(state);
    if (it == state_to_index.end())    return false;
    J = it->second; 
    return true;
}


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

void ZBasisHashmap::load_from_file(const fs::path& bfile, const std::string& dataset){
    this->ZBasisBase::load_from_file(bfile, dataset);
    build_index();
}

void ZBasisInterp::load_from_file(const fs::path& bfile, const std::string& dataset){
    this->ZBasisBase::load_from_file(bfile, dataset);
    find_bounds();
}

/*
void print_timings(const pthash::build_timings& timings){

    double total_microseconds = timings.partitioning_microseconds +
                                timings.mapping_ordering_microseconds +
                                timings.searching_microseconds + timings.encoding_microseconds;

    std::cout << "=== Construction time breakdown:\n";
    std::cout << "    partitioning: " << timings.partitioning_microseconds / 1000000.0
        << " [sec]"
        << " (" << (timings.partitioning_microseconds * 100.0 / total_microseconds)
        << "%)" << std::endl;
    std::cout << "    mapping+ordering: " << timings.mapping_ordering_microseconds / 1000000.0
        << " [sec]"
        << " (" << (timings.mapping_ordering_microseconds * 100.0 / total_microseconds)
        << "%)" << std::endl;
    std::cout << "    searching: " << timings.searching_microseconds / 1000000.0 << " [sec]"
        << " (" << (timings.searching_microseconds * 100.0 / total_microseconds) << "%)"
        << std::endl;
    //        std::cout << "    encoding: " << encoding_microseconds / 1000000.0 << " [sec]"
    //                  << " (" << (encoding_microseconds * 100.0 / total_microseconds) << "%)"
    //                  << std::endl;
    std::cout << "    total: " << total_microseconds / 1000000.0 << " [sec]" << std::endl;
}
*/

void ZBasisHashmap::build_index() {
    for (idx_t J=0; J<dim(); J++){
        state_to_index[states[J]]=J;
    }

//    // stage 1: construct the perfect hash fn
//    pthash::build_configuration config;
//    config.seed = 1234567890;
//    config.lambda = 5;
//    config.alpha = 0.97;
//    config.verbose = true;
//    config.avg_partition_size = 100000;
//    config.num_threads = 4;
//    config.dense_partitioning = true;
//
//    auto timings = phash.build_in_internal_memory(states.begin(), states.size(), config);
//    print_timings(timings);
//
//    std::vector<state_t> tmp_states;
//    tmp_states.resize(states.size());
//    std::swap(tmp_states, states);
//    std::cout <<"Original size "<<states.size() <<" phash size "<<phash.table_size()<<"\n";
//
//    idx_lookup.resize(phash.table_size());
//    // tmp_states now contains all the original states
//    for (idx_t J=0; J<dim(); J++){
//        auto state = tmp_states[J];
//        auto state_hash = phash(state);
//        states[state_hash] = state;
////        idx_lookup[state_hash] = J;
//    }
}


