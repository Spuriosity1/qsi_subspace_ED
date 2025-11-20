#pragma once
//#include "benes_network.hpp"
#include "operator.hpp"
#include <algorithm>
#include <set>
#include <vector>
#include <cassert>


struct qres_SymbolicPMROperator : public SymbolicPMROperator {

};



// the strategy: precompute all of these permuters ahead of time
template<typename state_t>
struct PermutationGroup {
    using perm_t = std::vector<ZBasisBase::idx_t>;

    PermutationGroup(std::vector<perm_t>& generators);

    void orbit(const state_t& psi, std::set<state_t>& orbit) const;
    void orbit(const state_t& psi, std::vector<state_t>& orbit) const;


    // repalces psi by its canonical representative in the G-orbit
    void make_representative(state_t& psi) const; 

    state_t get_representative(const state_t& state){
        state_t r = state;
        make_representative(r);
        return r;
    }

    size_t size() const { return group_elements.size(); }
    size_t dim() const { 
        return group_elements.size() == 0 ? 0 : group_elements[0].size();
    }

    perm_t operator[](size_t i) const {
        assert(i < this->size());
        return group_elements[i];
    }

//    auto get_permuter(size_t i ) const {
//        assert(i < this->size());
//        return permuters[i];
//    }

    protected:
    std::vector<perm_t> group_elements;
//    std::vector<BenesNetwork<state_t>> permuters;
    // Compose two permutations: result[i] = a[b[i]]
    static perm_t compose(const perm_t& a, const perm_t& b) {
        perm_t result(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[b[i]];
        }
        return result;
    }
};


template <RealOrCplx R, typename T>
struct Representation {
    Representation(const PermutationGroup<T>& G_, const std::vector<R> characters_) :
    G(G_), characters(characters_) {
        // make sure the convention of identity-first is respected
        for (size_t i=0; i<G.dim(); i++){
            assert(G[0][i] == i);
        }
    }
    const PermutationGroup<T>& G;
    const std::vector<R> characters;
    size_t dim() const {return characters[0]; 
        // trace of identity is dimension of the rep
    };
};

// Stores only one representative of each orbit, skipping states 
// entirely if they aren't present. Two parallel arrays:
// states -> the rerpresentatives, UInt128
// rep_norms -> doubles, rep_norms[i] is op-norm of P_{\Gamma} 
template <RealOrCplx R>
class GAdaptedZBasisBST : public ZBasisBST {
protected:
    const Representation<R, Uint128>& rep;
    std::vector<double> rep_norms; // norms of the basis vectors
    // meaning: rep_norms[i] == norm(P_{\chi} basis[i])

    // internal states buffer (inherited) is
    // now understood as containing only the canonical representative

    // calculates <beta_j | P_\gamma |beta_j>
    double calc_overlap(const state_t& psi){
        return 0;
    }

    // Finds representative states of each orbit, and calculates their overlap
    // with the specified representation
    void reduce_to_representatives(){

        rep_norms.resize(0);
        std::set<state_t> seen;
        std::vector<state_t> representatives;

        std::vector<state_t> curr_orbit;
        for (const auto& psi : states){
            // skip if we have seen psi before
            if (seen.contains(psi)) continue;
            // since states are sorted, we only need to push the first one we 
            // encounter (TODO verify)
            representatives.push_back(psi);

            rep.G.orbit(psi, curr_orbit);
            seen.insert(curr_orbit.begin(), curr_orbit.end());

            // build the operator
        }

        std::swap(representatives, this->states);
    }

public:
    GAdaptedZBasisBST(const PermutationGroup<Uint128>& G_, 
            const std::vector<R> chi) : rep(G_, chi){
    }
    GAdaptedZBasisBST(const Representation<R, Uint128>& R_) : rep(R_) {
    }

	void load_from_file(const fs::path& bfile, const std::string& dataset="basis"){
        // for now do this the stupid way: load the whole damn thing, then denude
        ZBasisBase::load_from_file(bfile, dataset);
        reduce_to_representatives();

    }

    int search(const state_t& state, idx_t& J) const {
        state_t chi = state;
        rep.G.make_representative(chi);
        return ZBasisBST::search(chi, J);
    }
    
};



template <typename T>
void PermutationGroup<T>::orbit(const T& psi, std::set<T>& orbit) const{
    orbit.clear();
    for (const auto& g : permuters){
        orbit.insert(g.apply(psi));
    }
}


template <typename T>
void PermutationGroup<T>::orbit(const T& psi, std::vector<T>& orbit) const{
    // up to roughly n=100, vector is faster than set
    orbit.clear();
    orbit.reserve(permuters.size());
    for (const auto& g : permuters){
        orbit.push_back(g.apply(psi));
    }
    std::sort(orbit.begin(), orbit.end());
    orbit.erase( std::unique(orbit.begin(), orbit.end()),  orbit.end());
}


template <typename T>
void PermutationGroup<T>::make_representative(T& psi) const {
    // up to roughly n=100, vector is faster than set
    T min = 0;
    min = ~min;
    for (const auto& g : permuters){
        auto chi = g.apply(psi);
        if (chi < min) {
            min = chi;
        }
    }
    psi = min;
}

