#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "operator_matrix.hpp"
#include <random>
#include "timeit.hpp"
#include <fstream>
#include "common_bits.hpp"
//#include "matrix_diag_bits.hpp"


using json = nlohmann::json;


int main(int argc, char* argv[]){
    
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file");
	prog.add_argument("-s", "--sector");
	prog.add_argument("--n_spinons")
        .default_value(0)
        .scan<'i', int>();

    prog.add_argument("--basis_file", "-b")
        .help("A basis file (HDF5 format). Defaults to ${lattice_file%.json}.h5");


    try {
        prog.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n";
        std::cerr << prog;
        return 1;
    }


	ZBasisBST basis;
   
	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;


    std::cout<<"[BST]  Loading basis..."<<std::endl;
    load_basis(basis, prog);
    std::cout<<"[BST]  Done! Basis dim="<<basis.dim()<<std::endl;


	using T=double;
	SymbolicOpSum<T> H_sym;

    std::vector<double> gv {1,1,1,1};

    auto [ringL, ringR, sl_list]  = get_ring_ops(jdata);

    size_t total_nnz=0;

    for (auto& O : ringL){
        for (size_t i=0; i<basis.dim(); i++){
            auto psi = basis[i];
            if (O.applyState(psi) != 0){
                total_nnz++;
            }
        }
    }

    printf("ringL: %zu/%zu nonvanishing (factor of %.2f)",
            total_nnz, basis.dim(), total_nnz * 1.0/ basis.dim() );

    return 0;
}
