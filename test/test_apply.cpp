#include "argparse/argparse.hpp"
#include "hamiltonian_setup.hpp"
#include <nlohmann/json.hpp>
#include "operator_matrix.hpp"
#include <random>
#include "timeit.hpp"



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

    std::cout<<"Loading basis..."<<std::endl;
	ZBasis basis;
   
	// Step 1: Load ring data from JSON
    auto lattice_file = prog.get<std::string>("lattice_file");
	std::ifstream jfile(lattice_file);
	if (!jfile) {
		std::cerr << "Failed to open JSON file\n";
		return 1;
	}
	json jdata;
	jfile >> jdata;

    load_basis(basis, prog);
    std::cout<<"Done! Basis dim="<<basis.dim()<<std::endl;


	using T=double;
	SymbolicOpSum<T> H_sym;
    
    std::vector<double> gv {1.0, -0.2, -0.2, -0.2};
    build_hamiltonian(H_sym, jdata, gv);

    auto H = LazyOpSum(basis, H_sym);

    std::vector<double> v, u;
    v.resize(basis.dim());
    u.resize(basis.dim());
    std::mt19937 rng(seed);
    set_random_unit(v, rng);

    TIMEIT("u <- Av", H.evaluate_add(v.data(), u.data());)
    return 0;
}
