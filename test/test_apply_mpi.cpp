#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "operator_matrix.hpp"
#include "operator_mpi.hpp"
#include <random>
#include "timeit.hpp"
#include "lanczos.hpp"
#include <fstream>
#include "basis_format_bits.hpp"




using json = nlohmann::json;


int main(int argc, char* argv[]){
    
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file");
	prog.add_argument("-s", "--sector");
	prog.add_argument("--n_spinons")
        .default_value(0)
        .scan<'i', int>();

    prog.add_argument("--seed")
        .help("Seed for the RNG")
        .scan<'i', unsigned int>()
        .default_value(0u);


    try {
        prog.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << prog;
        return 1;
    }


    unsigned int seed = prog.get<unsigned int>("--seed");
    
    MPI_Init(NULL, NULL);


	MPI_ZBasisBST basis;
    ZBasisBST basis_st;
//    ZBasisInterp basis_i;
   
	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;

    // Step 2: load and partition the basis
    std::cout<<"[MPI_BST]  Loading basis..."<<std::endl;
    MPIContext ctx = load_basis(basis, prog);
    std::cout<<"[MPI_BST]  Done! Basis dim="<<basis.dim()<<std::endl;

    std::cout<<"[BST]  Loading basis..."<<std::endl;
    load_basis(basis_st, prog);
    std::cout<<"[BST]  Done! Basis dim="<<basis_st.dim()<<std::endl;

	using T=double;
	SymbolicOpSum<T> H_sym;
    
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    auto H_mpi = MPILazyOpSum(basis, H_sym, ctx);
    auto H_st = LazyOpSum(basis_st, H_sym);

    std::vector<double> v_global, u_global, u_local;
    v_global.resize(basis_st.dim());
    u_global.resize(basis_st.dim());

    assert(ctx.local_block_size() == basis.dim());
    u_local.resize(ctx.local_block_size());

    std::mt19937 rng(seed);
    set_random_unit(v_global, rng);

    std::fill(u_global.begin(), u_global.end(), 0);
    std::fill(u_local.begin(), u_local.end(), 0);

    std::cout<<"[BST "<<ctx.world_rank<<"]  Apply..."<<std::endl;
    TIMEIT("u += Av", H_st.evaluate_add(v_global.data(), u_global.data());)

    std::cout<<"[BST_MPI "<<ctx.world_rank<<"]  Apply..."<<std::endl;
    // NOTE: add the local block offset to stay correct
    TIMEIT("u += Av", H_mpi.evaluate_add(v_global.data() + ctx.local_start_index(), u_local.data());)


    // we need to carefully check the offsets
    double tol =1e-9;
    auto start_offset = ctx.local_start_index();
    
    size_t error_count = 0;
    double max_error = 0;



    std::ostringstream filename;
    filename << "comparison_rank" << ctx.world_rank << ".csv";
    std::ofstream out(filename.str());
    out << "local_index,global_index,u_global,u_local\n";

    for (int i=0;  i<ctx.local_block_size(); i++){
        auto g_idx = start_offset + i;
        auto error = std::abs(u_global[g_idx] - u_local[i]);

        out << i << "," << g_idx << "," 
                << std::setprecision(17) << u_global[g_idx] << ","
                << std::setprecision(17) << u_local[i] << "\n";

        if( error > tol ){
            if (error_count == 0){
                std::cout<<"BST != MPI on global index "<< g_idx
                    <<"= ("<<ctx.world_rank<<") + "<<i<<": +"<<error<<"\n";
            }
            error_count++;
            max_error = std::max(max_error, error);
        }
    }

    if (error_count > 0) {
        std::cout << "Rank " << ctx.world_rank << ": " << error_count 
              << " errors found, max error = " << max_error << "\n";
    } else {
        std::cout << "Rank " << ctx.world_rank <<": All algos agree."<<std::endl;
    }
    MPI_Finalize();
    return 0;
}
