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

    auto sector_spec = prog.get<std::vector<int>>("--sector");

	std::string ext = ".";
	ext += std::to_string(prog.get<int>("n_spinon_pairs"));
	ext += prog.get<std::string>("extension");

    if (sector_spec.size() > 0){
        ext += "_s";
        bool dot=false;
        for (auto s : sector_spec){
            if (dot) ext += ".";
            ext += std::to_string(s);
            dot=true;
        }
        ext+=".partitioned";
    }


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
	prog.add_argument("--n_threads", "-t")
		.help("Number of threads to distribute across (works best as a power of 2)\
Setting to 0 will use the single-threaded implementations")
		.default_value(0)
		.scan<'i', int>();
	prog.add_argument("extension")
		.default_value(".basis");
	prog.add_argument("--order_spins")
		.choices("none", "greedy", "random")
		.default_value("greedy");
    prog.add_argument("--sector", "-s")
        .help("target global polarisation sector")
        .nargs(argparse::nargs_pattern::at_least_one)
        .default_value(std::vector<int>{})
        .scan<'d', int>();
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
	size_t n_threads = prog.get<int>("--n_threads");
	auto num_spinon_pairs = prog.get<int>("n_spinon_pairs");

    auto target_sector = prog.get<std::vector<int>>("--sector");

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
        if (target_sector.size() == 0){
            pyro_vtree<lat_container> L(lat, num_spinon_pairs);
            build_and_export(L, prog, perm);
        } else {
            cout <<"Looking for specific sector\n";
            pyro_vtree<lat_container_with_sector> L(lat, num_spinon_pairs);
            L.set_sector(target_sector);
            build_and_export(L, prog, perm);
        }
	} else {
        if (target_sector.size() == 0){
            pyro_vtree_parallel<lat_container> L(lat, num_spinon_pairs, n_threads);
            build_and_export(L, prog, perm);
        } else {
            
            cout <<"Looking for specific sector\n";
            pyro_vtree_parallel<lat_container_with_sector> L(lat, num_spinon_pairs, n_threads);
            L.set_sector(target_sector);
            build_and_export(L, prog, perm);
        }
	}	
	return 0;
}
