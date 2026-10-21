#pragma once
#include <cstdint>
#include "basis_cli.hpp"
#include "operator_mpi.hpp"

// MPI counterpart of with_serial_basis: instantiate the chosen hash-distributed
// basis wrapper (ZBasisMPI<...>) and hand it to a generic visitor. See
// basis_cli.hpp for the shared flag parsing and mask computation.
template<class Visitor>
int with_mpi_basis(BasisType t, uint64_t hi_mask, Visitor&& v) {
    switch (t) {
        case BasisType::BST:    { ZBasisBST_HashMPI     b;                      return v(b); }
        case BasisType::INTERP: { ZBasisInterp_HashMPI  b; b.set_hi_mask(hi_mask); return v(b); }
        case BasisType::FAST:   { ZBasisBSTFast_HashMPI b;                      return v(b); }
    }
    return 1;
}
