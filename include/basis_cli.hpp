#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <argparse/argparse.hpp>
#include "operator.hpp"

// Shared runtime selector for the local-basis search structure, so every
// executable exposes the same --basis-type / --interp-bits flags instead of
// hard-coding a `using basis_t = ...`. The three structures (ZBasisBST,
// ZBasisInterp, ZBasisBSTFast) are interchangeable implementations of the same
// search interface; see components/operator/include/operator.hpp.

enum class BasisType { BST, INTERP, FAST };

inline void provide_basis_options(argparse::ArgumentParser& prog) {
    prog.add_argument("--basis-type")
        .help("Basis search structure: bst | interp | fast")
        .choices("bst", "interp", "fast")
        .default_value(std::string("fast"));
    prog.add_argument("--interp-bits")
        .help("For --basis-type interp: number of top significant bits of the "
              "state (above spin 64) used as the bounds-map key. Default (-1) "
              "uses the top 18, computed from the spin count. Fewer bits = "
              "smaller map but wider per-key search range. Ignored otherwise.")
        .default_value(-1)
        .scan<'i', int>();
}

inline BasisType parse_basis_type(const argparse::ArgumentParser& prog) {
    const std::string bt = prog.get<std::string>("--basis-type");
    if (bt == "bst")    return BasisType::BST;
    if (bt == "interp") return BasisType::INTERP;
    if (bt == "fast")   return BasisType::FAST;
    throw std::runtime_error("Invalid --basis-type '" + bt +
                             "'. Must be one of: bst, interp, fast");
}

// Mask applied to a state's uint64[1] (bits 64..127) to key the interpolation
// bounds map. The populated hi bits for an n_spins system occupy positions
// 0..(n_spins-65); we retain the top `interp_bits` of those so the map has at
// most 2^interp_bits entries keyed on the highest-order state bits. interp_bits
// <= 0 selects the default of 18. For n_spins <= 64 every state shares
// uint64[1] == 0, so the mask is immaterial (return ~0ULL).
inline uint64_t compute_interp_hi_mask(int n_spins, int interp_bits) {
    const int sig = n_spins - 64;          // significant bits living in uint64[1]
    if (sig <= 0) return ~0ULL;
    int bits = (interp_bits <= 0) ? 18 : interp_bits;
    bits = std::min(bits, sig);
    const int shift = sig - bits;          // keep the TOP `bits` of the sig range
    const uint64_t low = (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1);
    return low << shift;
}

// Instantiate the chosen serial basis type and hand it to a generic visitor
// (e.g. a `[&](auto& basis) -> int { ... }` lambda). The switch lives here so
// each executable keeps a single body. hi_mask is only applied to ZBasisInterp.
template<class Visitor>
int with_serial_basis(BasisType t, uint64_t hi_mask, Visitor&& v) {
    switch (t) {
        case BasisType::BST:    { ZBasisBST     b;                      return v(b); }
        case BasisType::INTERP: { ZBasisInterp  b; b.set_hi_mask(hi_mask); return v(b); }
        case BasisType::FAST:   { ZBasisBSTFast b;                      return v(b); }
    }
    return 1;
}
