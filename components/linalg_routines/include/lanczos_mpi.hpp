#pragma once

#include "lanczos.hpp"
#include <mpi.h>
#include "operator_mpi.hpp"

namespace projED {
namespace lanczos_mpi {

    // specialise these in case we need extra knobs
struct Settings  : public lanczos::Settings 
    {
        Settings(const MPIContext& ctx_) : 
            ctx(ctx_){}
        const MPIContext& ctx;
    };

struct Result : public lanczos::Result 
    {};

inline std::ostream& operator<<(std::ostream& os, const Settings& settings)
{
    os << static_cast<const lanczos::Settings&>(settings);
    os << "---------------------------\n";
    os << "  Additional MPI config:\n";
    return os;
}


inline std::ostream& operator<<(std::ostream& os, const Result& res)
{
    os << static_cast<const lanczos::Result&>(res);
    os << "---------------------------\n";
    os << "  Additional MPI config:\n";
    return os;
}


// real and complex valued eigval0 routines.
// Note that the dimension of the Hilbert space is inferred from v.
Result eigval0(projED::RealApplyFn f, 
        double& eigval, 
        std::vector<double>& v, 
        const Settings& settings_
        );

Result eigval0(projED::ComplexApplyFn f, double& eigval, 
        std::vector<std::complex<double>>& v, 
        const Settings& settings_
        );


// end namespace lanczos
}
}
