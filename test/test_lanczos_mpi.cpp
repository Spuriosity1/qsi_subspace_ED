#include <cmath>
#include <iostream>
#include <vector>
#include "mpi.h"
#include "operator.hpp"
#include "lanczos_mpi.hpp"
#include <argparse/argparse.hpp>
#include <random>

using namespace projED;

//helper function
//
void generate_random_Herm(Eigen::MatrixXd& M, std::mt19937& rng){
    // Random Hermitian matrix
    std::normal_distribution<double> dist(0.0, 1.0);
    
    auto dim = M.cols();
    assert(M.cols() == M.rows());

    for (int i = 0; i < dim; i++) {
        for (int j = i; j < dim; j++) {
            double val = dist(rng);
            if (i == j)
                M(i,j) = val;          // diagonal
            else
                M(i,j) = M(j,i) = val; // symmetric
        }
    }
}

// ---------------- Main ----------------
int main(int argc, char** argv) {
    argparse::ArgumentParser program("lanczos_test");

    program.add_argument("--dim")
        .help("Matrix dimension")
        .scan<'i', int>()
        .default_value(100);

    program.add_argument("--krylov_dim", "-k")
        .help("Krylov space dimension")
        .scan<'i', int>()
        .default_value(30);

    program.add_argument("--max_iterations", "-M")
        .help("Max iterations before giving up")
        .scan<'i', int>()
        .default_value(5000);

    program.add_argument("--min_iterations", "-M")
        .help("Min iterations")
        .scan<'i', int>()
        .default_value(30);

    program.add_argument("--abs_tol", "-a")
        .help("Lanczos eigval atol e.g. -8 = 1e-8")
        .scan<'i', int>()
        .default_value(-8);

    program.add_argument("--rel_tol", "-r")
        .help("Lanczos eigval rtol e.g. -8 = 1e-8")
        .scan<'i', int>()
        .default_value(-8);

    program.add_argument("--seed")
        .help("Seed for the RNG")
        .scan<'i', unsigned int>()
        .default_value(0u);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << program;
        return 1;
    }

    MPI_Init(nullptr, nullptr);

    int dim = program.get<int>("--dim");
    unsigned int seed = program.get<unsigned int>("--seed");


    // construct the basis partition
    MPIctx ctx;
    ctx.partition_indices_equal(dim);

    // The random vector
    std::mt19937 rng(seed);

    Eigen::MatrixXd M(dim, dim);
    generate_random_Herm(M, rng);

    // Output vector
    std::vector<double> local_v0(ctx.local_block_size());

    lanczos_mpi::Settings settings(ctx);
    settings.krylov_dim = program.get<int>("--krylov_dim");
    settings.abs_tol = pow(10, program.get<int>("--abs_tol"));
    settings.rel_tol = pow(10, program.get<int>("--rel_tol"));

    settings.max_iterations = program.get<int>("--max_iterations");
    settings.min_iterations = program.get<int>("--min_iterations");

    settings.verbosity = 3;
    settings.calc_eigenvector = true;

       // Build size and displacement arrays for MPI collective operations
    std::vector<int> all_sizes(ctx.world_size), all_displs(ctx.world_size);
    for (int r = 0; r < ctx.world_size; ++r) {
        all_sizes[r] = static_cast<int>(ctx.block_size(r));
        all_displs[r] = static_cast<int>(ctx.idx_partition[r]);
    }

    using coeff_t = double;
    RealApplyFn evadd = [&M, &ctx, all_sizes, all_displs](const coeff_t* x_local, coeff_t* y_local){
        // For testing: gather x to all ranks, apply full matrix, extract local portion
        // In production: use distributed matrix storage
        auto dim = M.cols();
        
        // Get local size for this rank
        int local_size = static_cast<int>(ctx.local_block_size());
        int local_start = static_cast<int>(ctx.local_start_index());
        
        // Reconstruct global vector via Allgatherv
        Eigen::VectorXd x_global(dim);
        MPI_Allgatherv(x_local, local_size, MPI_DOUBLE,
                       x_global.data(), all_sizes.data(), all_displs.data(), 
                       MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Apply full matrix to get global result
        Eigen::VectorXd y_global = M * x_global;
        
        // Extract local portion and add to output
        for (int i = 0; i < local_size; ++i) {
            y_local[i] += y_global[local_start + i];
        }
    };
    double eigval_lanczos = 0.0;
    auto res = lanczos_mpi::eigval0(evadd, eigval_lanczos, local_v0, settings);



    // Exact solution with Eigen
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M);
    double eigval_exact = solver.eigenvalues()(0);
    auto eigenvector_exact = solver.eigenvectors().col(0);


    if (ctx.my_rank !=0){ 
        std::cout << "Lanczos smallest eigenvalue: " << eigval_lanczos << "\n";
        std::cout << "Exact   smallest eigenvalue: " << eigval_exact << "\n";
    }


    Eigen::Map<Eigen::VectorXd> v0_eigen(local_v0.data(), local_v0.size());


    // check rank-wise
    double err_eigvec = 0;

    
    auto err_eigval = std::abs(eigval_lanczos - eigval_exact);

    for (int i=0; i<ctx.local_block_size(); i++){
        double eps =  (local_v0[i] - eigenvector_exact[ctx.local_start_index() + i]);
        err_eigvec += eps * eps;
    }

    std::cout << "Eigenvalue error: " << err_eigval <<"\n";
    std::cout << "Eigenvector error: " << err_eigvec <<"\n";

    MPI_Finalize();

    if (!res.eigval_converged) {
        std::cout << "Test failed: Lanczos exceeded maximum iterations\n";
        return 1;
    } else if (err_eigval > 1e-4 ) {
        std::cout << "Test failed: eigval differs too much from exact result\n";
        return 2;
    } else if (err_eigvec > 1e-4 ) {
        std::cout << "Test failed: eigvec differs too much from exact result\n";
        return 3;
    } else { 
        std::cout << "Test passed\n";
        return 0; 
    }
}
