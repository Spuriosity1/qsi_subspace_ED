#include "pyro_tree.hpp"
#include "vanity.hpp"
#include <cstdio>
#include <fstream>
#include "admin.hpp"


using namespace std;
using json=nlohmann::json;


int main (int argc, char *argv[]) {
	if (argc < 3) {
		printf("USAGE: %s <latfile: json> <num_threads>\n", argv[0]);
	}

	std::string infilename(argv[1]);
	auto basis_file = as_basis_file(infilename);

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	pyro_vtree_parallel L(data, atoi(argv[2]));
	
	printf("Building state tree...\n");
	L.build_state_tree();
	
	FILE* outfile = std::fopen(basis_file.c_str(), "w");
	L.write_basis_file(outfile);
	std::fclose(outfile);

	return 0;
}
