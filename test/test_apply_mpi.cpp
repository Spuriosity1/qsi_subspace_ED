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


template<typename T>
void check_basis_partition(const T& basis, const MPIctx& ctx){

    std::cout<<"[MPI_BST] Checking basis partition..."<<std::endl;
        Uint128 state_prev=0;
        for (int il=0;  il<ctx.local_block_size(); il++){
            assert(basis[il] > state_prev);
            state_prev = basis[il];
            assert(basis[il] >= ctx.state_partition[ctx.my_rank]);
            assert(basis[il] < ctx.state_partition[ctx.my_rank+1]);
        }
    std::cout<<"Done"<<std::endl;
}




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

    prog.add_argument("--trim")
        .default_value(false)
        .implicit_value(true);
    prog.add_argument("--rebalance")
        .default_value(false)
        .implicit_value(true);


    try {
        prog.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << prog;
        return 1;
    }


    unsigned int seed = prog.get<unsigned int>("--seed");
    
    MPI_Init(NULL, NULL);


	ZBasisBST_MPI basis;

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
    SparseMPIContext ctx = load_basis(basis, prog);
    std::cout<<"[MPI_BST]  Done! Basis dim="<<basis.dim()<<std::endl;

    std::cout<<"[BST]  Loading basis..."<<std::endl;
    load_basis(basis_st, prog);
    std::cout<<"[BST]  Done! Basis dim="<<basis_st.dim()<<std::endl;

	using T=double;
	SymbolicOpSum<T> H_sym;

    
    
    auto [ringL, ringR, sl_list]  = get_ring_ops(jdata);
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    for (size_t idx=0; idx<sl_list.size(); idx++){
        auto R = ringR[idx];
        auto L = ringL[idx];

        H_sym.add_term(gv[sl_list[idx]], R);
        H_sym.add_term(gv[sl_list[idx]], L);
    }


    ctx.log<<"[Symbolic ham construction done.]"<<std::endl;


    {
        std::cout<<"[MPI_BST] Checking applyState consistency..."<<std::endl;

        for (auto& [c, O] : H_sym.off_diag_terms){
            for (int il=0;  il<ctx.local_block_size(); il++){
                auto p1 = basis[il];
                auto p2 = basis[il];

                int s1 = O.applyState(p1);
                int s2 = O.applyState_branch(p2);
                assert(s1 == s2);
                if (s1 != 0){
                    assert(p1 == p2);
                }
            }
        }

        std::cout<<"Done"<<std::endl;
    }



    if (prog.get<bool>("--trim")){
        ctx.log<<"[remove unneeded elements]"<<std::endl;
        basis_st.remove_null_states(H_sym);
        basis.remove_null_states(H_sym, ctx);
    }
 
    ctx.log<<"[op construct]"<<std::endl;



    auto H_mpi_batch = MPILazyOpSumBatched<double, ZBasisBST_MPI>(basis, H_sym, ctx);
    auto H_mpi_pipe = MPILazyOpSumPipe<double, ZBasisBST_MPI>(basis, H_sym, ctx);
    auto H_mpi_pipeP = MPILazyOpSumPipePrealloc<double, ZBasisBST_MPI>(basis, H_sym, ctx);

    auto H_st = LazyOpSum(basis_st, H_sym);


    if (prog.get<bool>("--rebalance")){
        ctx.log<<"[calc basis wisdom]"<<std::endl;
        auto wisdom = H_mpi_pipe.find_optimal_basis_load();
        ctx.log<<"[basis reshuffle]"<<std::endl;
        basis.exchange_local_states(wisdom, ctx);
    }

    ctx.log<<"[allocate temporaries]"<<std::endl;
    H_mpi_pipe.allocate_temporaries();
    H_mpi_pipeP.allocate_temporaries();
    H_mpi_batch.allocate_temporaries();

    std::vector<double> v_global, u_global;
    std::array<std::vector<double>, 3> u_local;

    v_global.resize(basis_st.dim());
    u_global.resize(basis_st.dim());
    assert(ctx.local_block_size() == basis.dim());
    for (auto& ul : u_local){
        ul.resize(ctx.local_block_size());
    }
//    u2_local.resize(ctx.local_block_size());

    std::mt19937 rng(seed);
    projED::set_random_unit(v_global, rng);

    std::fill(u_global.begin(), u_global.end(), 0);
    for (auto& ul : u_local){
        std::fill(ul.begin(), ul.end(), 0);
    }
//    std::fill(u2_local.begin(), u2_local.end(), 0);


    check_basis_partition(basis, ctx);

    std::cout<<"[BST "<<ctx.my_rank<<"]  Apply..."<<std::endl;
    TIMEIT("[BST] u += Av", H_st.evaluate_add(v_global.data(), u_global.data());)

    std::cout<<"[BST_MPI "<<ctx.my_rank<<"]  Apply..."<<std::endl;
    // NOTE: add the local block offset to stay correct
    TIMEIT("[MPI batch] u += Av", H_mpi_batch.evaluate_add(v_global.data() + ctx.local_start_index(), u_local[0].data());)
    TIMEIT("[MPI pipe] u += Av", H_mpi_pipe.evaluate_add(v_global.data() + ctx.local_start_index(), u_local[1].data());)
    TIMEIT("[MPI pipe prealloc] u += Av", H_mpi_pipeP.evaluate_add(v_global.data() + ctx.local_start_index(), u_local[2].data());)


    std::vector<std::string> names = {"MPI batch", "MPI pipe", "MPI pipe prealloc"};

    // we need to carefully check the offsets
    double tol =1e-9;
    auto start_offset = ctx.local_start_index();
    

    std::array<double, 3> error;
    std::array<double, 3> max_error = {0,0,0};
    std::array<int, 3> error_count = {0,0,0};


    for (int mu=0; mu<3; mu++){
        for (int i=0;  i<ctx.local_block_size(); i++){
            auto g_idx = start_offset + i;
            error[mu] = std::abs(u_global[g_idx] - u_local[mu][i]);

            if( error[mu] > tol ){
                if (error_count[mu] == 0){
                    std::cout<<"BST != "<<names[mu]<<" on global index "<< g_idx
                        <<"= (rank "<<ctx.my_rank<<") + "<<i<<": +"<<error[mu]<<
                        "\t"<<"Expected "<<u_global[g_idx]<<" got "<<u_local[mu][i]<<"\n";
                }
                error_count[mu]++;
                max_error[mu] = std::max(max_error[mu], error[mu]);
            }
        }

        if (error_count[mu] > 0) {
            std::cout << "["<<names[mu]<<"] Rank " << ctx.my_rank << ": " << error_count[mu] 
                  << " errors found, max error = " << max_error[mu] << "\n";
        } else {
            std::cout << "["<<names[mu]<<"] Rank " << ctx.my_rank <<": agrees with global BST."<<std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
