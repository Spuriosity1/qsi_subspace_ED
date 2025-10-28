#include "argparse/argparse.hpp"
#include "pyro_tree.hpp"
#include "vanity.hpp"
#include <cstdio>
#include <fstream>
#include "admin.hpp"


// Enumerates all states stisfying local constraints using truncated binary search.
// Builds all states in memory, do not use for large problems.

using namespace std;
using json=nlohmann::json;


template <typename T>
void build_and_export(T& L, const argparse::ArgumentParser& prog,
		const std::vector<size_t>& perm){

	std::string ext = ".";
	ext += std::to_string(prog.get<int>("n_spinon_pairs"));
	ext += prog.get<std::string>("extension");

	auto outfilename=as_basis_file(prog.get<std::string>("lattice_file"), ext );

	printf("Building state tree...\n");
	L.build_state_tree();

	// undo the permutation from earlier to keep the output order consistent
	printf("Permuting...\n");
	L.permute_spins(perm);

	printf("Sorting...\n");
	L.sort();
	
    char out_fmt = prog.get<std::string>("--out_format")[0];
#ifdef DONT_USE_HDF5
    out_fmt = 'c';
#endif
	switch(out_fmt){
		case 'c': // csv 
	        printf("[csv] Saving...\n");
			L.write_basis_csv(outfilename);
			break;
		case 'h': // h5 
	        printf("[h5] Saving...\n");
			L.write_basis_hdf5(outfilename);
			break;
		case 'n': // do not save (why whould you want this?)
	        printf("Not saving...\n");
			break;
		default: // also catches 'both' case 
	        printf("[csv] Saving...\n");
			L.write_basis_csv(outfilename);
	        printf("[h5] Saving...\n");
			L.write_basis_hdf5(outfilename);
	}

}


int main (int argc, char *argv[]) {
	argparse::ArgumentParser prog("gen_spinon_basis");
	prog.add_argument("lattice_file")
		.help("The json-vlaued lattice spec");
	prog.add_argument("n_spinon_pairs")
		.default_value(0)
		.scan<'i', int>();
	prog.add_argument("--n_threads")
		.help("Number of threads to distribute across (works best as a power of 2)\
Setting to 0 will use the single-threaded implementations")
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
	size_t n_threads = prog.get<int>("n_threads");
	auto num_spinon_pairs = prog.get<int>("n_spinon_pairs");

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
	
	if (n_threads == 0){
		pyro_vtree L(lat, num_spinon_pairs);
		build_and_export(L, prog, perm);
	} else {
		pyro_vtree_parallel L(lat, num_spinon_pairs, n_threads);
		build_and_export(L, prog, perm);
	}	
	return 0;
}
