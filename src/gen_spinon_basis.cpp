#include "pyro_tree.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include "admin.hpp"


using namespace std;
using json=nlohmann::json;


int main (int argc, char *argv[]) {
	if (argc < 2) {
		printf("USAGE: %s <latfile: json> (<n_spinon_pairs>=0) (<ext>=.basis)\n", argv[0]);
		return 1;
	}

	std::string infilename(argv[1]);

	unsigned num_spinon_pairs=(argc >= 3) ? atoi(argv[2]) : 0;

	std::string ext = ".";
	ext += std::to_string(num_spinon_pairs);
	ext += (argc >= 4) ? argv[3] : ".basis";

	auto outfilename=as_basis_file(infilename, ext );

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	basis_tree_builder L(data, num_spinon_pairs);
	
	printf("Building state tree...\n");
	L.build_state_tree();

	printf("Sorting...\n");
	L.sort();
	
	L.write_basis_csv(outfilename);
	L.write_basis_hdf5(outfilename);

	return 0;
}
