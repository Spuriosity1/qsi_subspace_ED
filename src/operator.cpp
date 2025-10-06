#include "operator.hpp"
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
//

bool ZBasis::search(const state_t& state, idx_t& J) const {
//    auto it = std::lower_bound(states.begin(), states.end(), state);
//    return it;
    size_t left = 0, right = states.size() - 1;
    
    while (right - left > 64) {
        size_t mid = (left + right) / 2;
//        size_t prefetch_left = (left + mid) / 2;
//        size_t prefetch_right = (mid + right) / 2;
        
//        __builtin_prefetch(&states[prefetch_left], 0, 0);
//        __builtin_prefetch(&states[prefetch_right], 0, 0);
        
        if (states[mid] < state) left = mid + 1;
        else right = mid;
    }

    for (J = left; J <= right; J++) {
        if (states[J] == state) return true;
    }
    return false; // not found;
}

//ZBasis::idx_t ZBasis::idx_of_state(const ZBasis::state_t& state) const {
//    auto it = find(state);
//#ifdef DEBUG 
//    if (it == state.end() || *it != state){
//        throw state_not_found_error(state);
//    }
//#endif
//    return std::distance(states.begin(), it);
//}


size_t ZBasis::insert_states(const std::vector<ZBasis::state_t>& to_insert,
        std::vector<ZBasis::state_t>& new_states){
    new_states.resize(0);
    size_t n_insertions = 0;
    for (auto& s : to_insert){
        ZBasis::idx_t tmp;
        // skip if we know about it already
        if (this->search(s, tmp)) continue;
//        state_to_index[s] = states.size();
        states.push_back(s);
        new_states.push_back(s);
        n_insertions++;
    }
    return n_insertions;
}

void ZBasis::load_from_file(const fs::path& bfile, const std::string& dataset){
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

//    for (idx_t i=0; i<states.size(); i++){
//        state_to_index[states[i]] = i;
//    }
}


