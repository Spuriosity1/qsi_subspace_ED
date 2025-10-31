#pragma once
#include <argparse/argparse.hpp>


inline void provide_lanczos_options(argparse::ArgumentParser& prog){
	prog.add_argument("--ncv", "-k")
		.help("Krylov dimension, should be > 2*n_eigvals")
		.default_value(15)
		.scan<'i', int>();
	prog.add_argument("--atol")
		.help("Absolute convergence tolerance, specify as exponent e.g. -8 = 10^-8")
		.default_value(-8)
		.scan<'i', int>();
	prog.add_argument("--rtol")
		.help("Relative tolerance, specify as exponent e.g. -8 = 10^-8")
		.default_value(-8)
		.scan<'i', int>();
	prog.add_argument("--rng_seed", "-s")
		.help("Seed used to generate random Lanczos vectors.")
		.default_value(0)
		.scan<'i', int>();
    prog.add_argument("--verify")
        .help("Flag to make lanczos check that its answer is right")
		.default_value(false)
		.implicit_value(true);
}

template<typename T>
inline void parse_lanczos_settings(const argparse::ArgumentParser& prog, 
        T& sett){
    sett.krylov_dim = prog.get<int>("--ncv");
    sett.verbosity = 2;
    sett.min_iterations = 10;
    sett.calc_eigenvector = true;
    sett.x0_seed = prog.get<int>("--rng_seed");
    sett.abs_tol = pow(10,prog.get<int>("--atol"));
    sett.rel_tol = pow(10,prog.get<int>("--rtol"));
    sett.verify_eigenvector = prog.get<bool>("--verify");
}
