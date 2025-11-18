#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "operator_matrix.hpp"
#include "operator_mpi.hpp"
#include <random>
#include "timeit.hpp"
//#include "lanczos.hpp"
#include "common_bits.hpp"
#include <fstream>
#include "basis_format_bits.hpp"




using json = nlohmann::json;


int main(int argc, char* argv[]){
    
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file");
    prog.add_argument("--basis_file", "-b")
        .help("A basis file (HDF5 format). Defaults to ${lattice_file%.json}.h5");
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
    
//    build_hamiltonian(H_sym, jdata, gv);
    
    auto [ringL, ringR, sl_list]  = get_ring_ops(jdata);
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
//    std::vector<int> indices{8,8};
    for (size_t idx=0; idx<sl_list.size(); idx++){
//    for (auto idx : indices){
        auto R = ringR[idx];
        auto L = ringL[idx];

        H_sym.add_term(gv[sl_list[idx]], R);
        H_sym.add_term(gv[sl_list[idx]], L);
    }
 

    auto H_mpi = MPILazyOpSum(basis, H_sym, ctx);
    auto H_st = LazyOpSum(basis_st, H_sym);

    std::vector<double> v_global, u_global, u1_local;
    v_global.resize(basis_st.dim());
    u_global.resize(basis_st.dim());

    assert(ctx.local_block_size() == basis.dim());
    u1_local.resize(ctx.local_block_size());
//    u2_local.resize(ctx.local_block_size());

    std::mt19937 rng(seed);
    projED::set_random_unit(v_global, rng);

    std::fill(u_global.begin(), u_global.end(), 0);
    std::fill(u1_local.begin(), u1_local.end(), 0);
//    std::fill(u2_local.begin(), u2_local.end(), 0);

    std::cout<<"[BST "<<ctx.my_rank<<"]  Apply..."<<std::endl;
    TIMEIT("[BST] u += Av", H_st.evaluate_add(v_global.data(), u_global.data());)

    std::cout<<"[BST_MPI "<<ctx.my_rank<<"]  Apply..."<<std::endl;
    // NOTE: add the local block offset to stay correct
    TIMEIT("[MPI] u += Av", H_mpi.evaluate_add(v_global.data() + ctx.local_start_index(), u1_local.data());)

    // we need to carefully check the offsets
    double tol =1e-9;
    auto start_offset = ctx.local_start_index();
    
    size_t error_1_count = 0;
    double max_error_1 = 0;
//    size_t error_2_count = 0;
//    double max_error_2 = 0;



    std::ostringstream filename;
    filename << "comparison_rank" << ctx.my_rank << ".csv";
    std::ofstream out(filename.str());
    out << "local_index,global_index,u_global,u_local\n";

    for (int i=0;  i<ctx.local_block_size(); i++){
        auto g_idx = start_offset + i;
        auto error_1 = std::abs(u_global[g_idx] - u1_local[i]);
//        auto error_2 = std::abs(u_global[g_idx] - u2_local[i]);

        if( error_1 > tol ){
            if (error_1_count == 0){
                std::cout<<"BST != MPI sync on global index "<< g_idx
                    <<"= ("<<ctx.my_rank<<") + "<<i<<": +"<<error_1<<"\n";
            }
            error_1_count++;
            max_error_1 = std::max(max_error_1, error_1);
        }


//        if( error_2 > tol ){
//            if (error_2_count == 0){
//                std::cout<<"BST != MPI pipe on global index "<< g_idx
//                    <<"= ("<<ctx.my_rank<<") + "<<i<<": +"<<error_2<<"\n";
//            }
//            error_2_count++;
//            max_error_2 = std::max(max_error_2, error_2);
//        }
    }

    if (error_1_count > 0) {
        std::cout << "[MPI] Rank " << ctx.my_rank << ": " << error_1_count 
              << " errors found, max error = " << max_error_1 << "\n";
    } else {
        std::cout << "[MPI] Rank " << ctx.my_rank <<": agrees with global BST."<<std::endl;
    }


//    if (error_2_count > 0) {
//        std::cout << "[pipe] Rank " << ctx.my_rank << ": " << error_2_count 
//              << " errors found, max error = " << max_error_2 << "\n";
//    } else {
//        std::cout << "[pipe] Rank " << ctx.my_rank <<": agrees with global BST."<<std::endl;
//    }
    MPI_Finalize();
    return 0;
}
