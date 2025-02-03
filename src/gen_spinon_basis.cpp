#include "pyro_tree.hpp"
#include <cstdio>
#include <fstream>
#include "admin.hpp"


using namespace std;
using json=nlohmann::json;


int main (int argc, char *argv[]) {
	if (argc < 2) {
		printf("USAGE: %s <latfile: json\n", argv[0]);
	}

	std::string infilename(argv[1]);
	string outfilename=as_basis_file(infilename);

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	pyro_vtree L(data);
	
	printf("Building state tree...\n");
	L.build_state_tree();
	
	FILE* outfile = std::fopen(outfilename.c_str(), "w");
	L.write_basis_file(outfile);
	std::fclose(outfile);

	return 0;
}
