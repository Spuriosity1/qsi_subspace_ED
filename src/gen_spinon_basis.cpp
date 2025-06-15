#include "pyro_tree.hpp"
#include <cstdio>
#include <fstream>
#include <random>
#include <argparse/argparse.hpp>
#include <string>
#include "admin.hpp"


using namespace std;
using json=nlohmann::json;


int main (int argc, char *argv[]) {
	argparse::ArgumentParser prog("gen_spinon_basis");
	prog.add_argument("lattice_file")
		.help("The json-vlaued lattice spec");
	prog.add_argument("n_spinon_pairs")
		.default_value(0)
		.scan<'i', int>();
	prog.add_argument("extension")
		.default_value(".basis");
	prog.add_argument("--order_spins")
		.choices("none", "greedy", "random")
		.default_value("greedy");


    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
		std::cerr << err.what() << std::endl;
		std::cerr << prog;
        std::exit(1);
    }

	std::string infilename = prog.get<std::string>("lattice_file");
	unsigned num_spinon_pairs= prog.get<int>("n_spinon_pairs");

	std::string ext = ".";
	ext += std::to_string(num_spinon_pairs);
	ext += prog.get<std::string>("extension");

	auto outfilename=as_basis_file(infilename, ext );

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	lattice lat(data);


	std::vector<size_t> perm;	
	for (size_t i=0; i<lat.spins.size(); i++){
		perm.push_back(i);
	}

	auto choice = prog.get<std::string>("--order_spins");

	std::random_device rd;
	std::mt19937 rng(rd());
	if (choice == "greedy"){
		perm = lat.greedy_spin_ordering(0);
	} else if (choice == "random") {
		std::shuffle(perm.begin(), perm.end(), rng);
	}

	lat.apply_permutation(perm);

	pyro_vtree L(lat, num_spinon_pairs);
	
	printf("Building state tree...\n");
	L.build_state_tree();

	L.permute_spins((perm));

	printf("Sorting...\n");
	L.sort();
	
	L.write_basis_csv(outfilename);
	L.write_basis_hdf5(outfilename);

	return 0;
}
