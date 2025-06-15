#include "pyro_tree.hpp"
#include <cstdio>
#include <fstream>
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
	prog.add_argument("--out_format")
		.choices("csv", "h5", "both", "none")
		.default_value("h5");


    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
		std::cerr << err.what() << std::endl;
		std::cerr << prog;
        std::exit(1);
    }

	auto infilename = prog.get<std::string>("lattice_file");
	auto num_spinon_pairs= prog.get<int>("n_spinon_pairs");

	std::string ext = ".";
	ext += std::to_string(num_spinon_pairs);
	ext += prog.get<std::string>("extension");

	auto outfilename=as_basis_file(infilename, ext );

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	// Parameter parsing complete. Load lattice into a container struct

	lattice lat(data);

	auto choice = prog.get<std::string>("--order_spins");

	// Optionally permute the indices to make the early tree
	// truncation as efficient as possible. We permute indices, run the algo,
	// then finally unpermute when saving to file
	std::vector<size_t> perm = get_permutation(choice, lat);

	lat.apply_permutation(perm);

	pyro_vtree L(lat, num_spinon_pairs);
	
	printf("Building state tree...\n");
	L.build_state_tree();
	
	// undo the permutation from earlier to keep the output order consistent
	L.permute_spins(perm);

	printf("Sorting...\n");
	L.sort();
	
	switch(prog.get<std::string>("--out_format")[0]){
		case 'c': // csv
			L.write_basis_csv(outfilename);
			break;
		case 'h': // h5 
			L.write_basis_hdf5(outfilename);
			break;
		case 'n': // do not save (why whould you want this?)
			break;
		default: // also catches 'both' case
			L.write_basis_csv(outfilename);
			L.write_basis_hdf5(outfilename);
	}

	return 0;
}
