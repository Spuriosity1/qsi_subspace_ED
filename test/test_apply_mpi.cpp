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


//template<typename T>
//void check_basis_partition(const T& basis, const MPIctx& ctx){
//
//    std::cout<<"[MPI_BST] Checking basis partition..."<<std::endl;
//        Uint128 state_prev=0;
//        for (int il=0;  il<ctx.local_block_size(); il++){
//            assert(basis[il] > state_prev);
//            state_prev = basis[il];
//            assert(basis[il] >= ctx.state_partition[ctx.my_rank]);
//            assert(basis[il] < ctx.state_partition[ctx.my_rank+1]);
//        }
//    std::cout<<"Done"<<std::endl;
//}




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

    prog.add_argument("--notrim")
        .default_value(false)
        .implicit_value(true);
//    prog.add_argument("--rebalance")
//        .default_value(false)
//        .implicit_value(true);


    try {
        prog.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << prog;
        return 1;
    }


    unsigned int seed = prog.get<unsigned int>("--seed");
    
    MPI_Init(NULL, NULL);


	ZBasisBST_HashMPI     basis_loc;
    ZBasisBSTFast_HashMPI basis_fast_loc;
    ZBasisBST basis_st;
   
	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;

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

    // Step 2: load raw slab, trim, then redistribute to hash-correct ranks
    std::cout<<"[MPI_BST]  Loading basis..."<<std::endl;
    load_basis_raw(basis_loc, prog);
    std::cout<<"[MPI_fast] Loading basis..."<<std::endl;
    load_basis_raw(basis_fast_loc, prog);
    std::cout<<"[BST]  Loading basis..."<<std::endl;
    load_basis(basis_st, prog);

    MPIHashContext ctx;

    if (!prog.get<bool>("--notrim")){
        ctx.log<<"[remove unneeded elements]"<<std::endl;
        basis_st.remove_null_states(H_sym);
        basis_loc.remove_null_states(H_sym);
        basis_fast_loc.remove_null_states(H_sym);
    }

    basis_loc.redistribute();
    std::cout<<"[MPI_BST]  Done! Basis dim="<<basis_loc.dim()<<std::endl;
    basis_fast_loc.redistribute();
    std::cout<<"[MPI_fast] Done! Basis dim="<<basis_fast_loc.dim()<<std::endl;
    std::cout<<"[BST]  Done! Basis dim="<<basis_st.dim()<<std::endl;

    std::cout<<"[MPI_BST] Checking applyState consistency..."<<std::endl;

    for (auto& [c, O] : H_sym.off_diag_terms){
        for (int il=0;  il<basis_loc.dim(); il++){
            auto p1 = basis_loc[il];
            auto p2 = basis_loc[il];

            int s1 = O.applyState(p1);
            int s2 = O.applyState_branch(p2);
            assert(s1 == s2);
            if (s1 != 0){
                assert(p1 == p2);
            }
        }
    }

    std::cout<<"Done"<<std::endl;

    ctx.log<<"[Symbolic ham construction done.]"<<std::endl;
 
    ctx.log<<"[op construct]"<<std::endl;
    auto H_mpi  = MPILazyOpSum(basis_loc,      H_sym, ctx);
    auto H_fast = MPILazyOpSum(basis_fast_loc, H_sym, ctx);
    auto H_st   = LazyOpSum(basis_st, H_sym);


//    if (prog.get<bool>("--rebalance")){
//        ctx.log<<"[calc basis wisdom]"<<std::endl;
//        auto wisdom = H_mpi.find_optimal_basis_load();
//        ctx.log<<"[basis reshuffle]"<<std::endl;
//        basis.exchange_local_states(wisdom, ctx);
//    }
    ctx.log<<"[allocate temporaries]"<<std::endl;
    H_mpi.allocate_temporaries();
    H_fast.allocate_temporaries();

    std::vector<double> v_global, v_local, u_global, u1_local, u2_local;
    v_global.resize(basis_st.dim());
    u_global.resize(basis_st.dim());

    u1_local.resize(basis_loc.dim());
    u2_local.resize(basis_fast_loc.dim());


    std::mt19937 rng(seed);
    projED::set_random_unit(v_global, rng);

    // populate v_local
    v_local.reserve(basis_loc.dim());
    std::vector<int64_t> index_map;
    for (int64_t ig=0; ig<basis_st.dim(); ig++){
        const auto& psi = basis_st[ig];
        if (ctx.rank_of_state(psi)== ctx.my_rank){
            index_map.push_back(ig);
            v_local.push_back(v_global[ig]);
        }
    }

    std::fill(u_global.begin(), u_global.end(), 0);
    std::fill(u1_local.begin(), u1_local.end(), 0);
    std::fill(u2_local.begin(), u2_local.end(), 0);

    std::cout<<"[BST "<<ctx.my_rank<<"]  Apply..."<<std::endl;
    TIMEIT("[BST]      u += Av", H_st.evaluate_add(v_global.data(), u_global.data());)

    std::cout<<"[BST_MPI "<<ctx.my_rank<<"]  Apply..."<<std::endl;
    TIMEIT("[MPI_BST]  u += Av", H_mpi.evaluate_add(v_local.data(), u1_local.data());)

    std::cout<<"[fast_MPI "<<ctx.my_rank<<"] Apply..."<<std::endl;
    TIMEIT("[MPI_fast] u += Av", H_fast.evaluate_add(v_local.data(), u2_local.data());)

    double tol = 1e-9;
    bool ok = true;

    auto check = [&](const char* label, const std::vector<double>& u_loc) {
        size_t nerr = 0;
        double max_err = 0;
        for (int i = 0; i < (int)u_loc.size(); i++){
            auto g_idx = index_map[i];
            auto err = std::abs(u_global[g_idx] - u_loc[i]);
            if (err > tol){ nerr++; max_err = std::max(max_err, err); }
        }
        if (nerr > 0){
            std::cout << label << " rank " << ctx.my_rank << ": "
                      << nerr << " errors, max=" << max_err << "\n";
            ok = false;
        } else {
            std::cout << label << " rank " << ctx.my_rank << ": agrees with global BST.\n";
        }
    };

    check("[MPI_BST] ", u1_local);
    check("[MPI_fast]", u2_local);

    MPI_Finalize();
    return ok ? 0 : 1;
}
