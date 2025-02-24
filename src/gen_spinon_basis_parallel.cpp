#include "pyro_tree.hpp"
#include "vanity.hpp"
#include <cstdio>
#include <fstream>
#include "admin.hpp"


using namespace std;
using json=nlohmann::json;


int main (int argc, char *argv[]) {
	if (argc < 3) {
		printf("USAGE: %s <latfile: json> <num_threads> (<num_spinon_pairs>=0 <ext>=.basis)\n", argv[0]);
	}

	std::string infilename(argv[1]);
	int num_spinon_pairs=(argc >= 4) ? atoi(argv[3]) : 0;

	std::string ext = ".";
	ext += std::to_string(num_spinon_pairs);
	ext += (argc >= 5) ? argv[4] : ".basis";

	auto basis_file = as_basis_file(infilename,ext );

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	pyro_vtree_parallel L(data, num_spinon_pairs, atoi(argv[2]));
	
	printf("Building state tree...\n");
	L.build_state_tree();
	
	FILE* outfile = std::fopen(basis_file.c_str(), "w");
	L.write_basis_file(outfile);
	std::fclose(outfile);

	return 0;
}
