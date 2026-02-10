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

    prog.add_argument("--trim")
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


	MPI_ZBasisBST basis;
   
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


    if (prog.get<bool>("--trim")){
        ctx.log<<"[remove unneeded elements]"<<std::endl;
        basis.remove_null_states(H_sym, ctx);
    }

    assert(ctx.local_block_size() == basis.size());

    
//    const auto lower_bound = ctx.state_partition[ctx.my_rank];
//    const auto upper_bound = ctx.state_partition[ctx.my_rank+1];
    for (int il=0; il<basis.size(); il++){
//        if(basis[il] < lower_bound){
//            std::cerr <<"[rank "<<ctx.my_rank<<"] State "<<il<<" = "<<basis[il] << " too small | should be >= "<<lower_bound<<std::endl;
//            throw std::logic_error("bad partition!");
//        } else if(basis[il] >= upper_bound){
//            std::cerr <<"[rank "<<ctx.my_rank<<"] State "<<il<<" = "<<basis[il] << " too big | should be < "<<upper_bound<<std::endl;
//            throw std::logic_error("bad partition!");
//        }
        assert(ctx.rank_of_state(basis[il]) == ctx.my_rank);
    }

    MPI_Finalize();
    return 0;
}
