#include "argparse/argparse.hpp"
#include "pyro_tree_mpi.hpp"
#include "vanity.hpp"
#include <cstdio>
#include <fstream>
#include "admin.hpp"
#include <mpi.h>



// a more low level gen_spinon_basis.
// Writes streamed, sharded data to temporary files
// Sharded Binary SEARCH implementation
// Collect all data together with a final call to 

using namespace std;
using json=nlohmann::json;

int main (int argc, char *argv[]) {
	argparse::ArgumentParser prog(argv[0]);
	prog.add_argument("lattice_file")
		.help("The json-vlaued lattice spec");
	prog.add_argument("n_spinon_pairs")
		.default_value(0)
		.scan<'i', int>();
	prog.add_argument("--order_spins")
		.choices("none", "greedy", "random")
		.default_value("greedy");
	prog.add_argument("--tmp_outpath", "-o")
        .required()
        .help("The directory to push sharded output to");
    prog.add_argument("--buffer_size", "-b")
        .help("RAM buffer size for the state tree")
        .default_value(1<<20)
        .scan<'i', int>();

    try {
        prog.parse_args(argc, argv);
    } catch (const std::exception& err){
		std::cerr << err.what() << std::endl;
		std::cerr << prog;
        std::exit(1);
    }

	auto infilename = prog.get<std::string>("lattice_file");
	auto num_spinon_pairs = prog.get<int>("n_spinon_pairs");

    auto tmp_outpath = prog.get<std::string>("--tmp_outpath");
    auto buffer_size = prog.get<int>("--buffer_size");

    std::filesystem::path inpath(infilename);


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

    MPI_Init(nullptr, nullptr);
	
    mpi_par_searcher L(lat, num_spinon_pairs, perm,
            tmp_outpath,
            inpath.stem(),
            buffer_size);

    L.build_state_tree();
    L.finalise_shards();
    MPI_Finalize();
	
	return 0;
}
