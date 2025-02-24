#include "pyro_tree.hpp"
#include <cstdio>
#include <fstream>
#include "admin.hpp"


using namespace std;
using json=nlohmann::json;


int main (int argc, char *argv[]) {
	if (argc < 2) {
		printf("USAGE: %s <latfile: json> (<n_spinon_pairs>=0) (<ext>=.basis)\n", argv[0]);
	}

	std::string infilename(argv[1]);
	std::string outfilename;

	int num_spinon_pairs=(argc >= 3) ? atoi(argv[2]) : 0;

	outfilename=as_basis_file(infilename, (argc >= 4) ? argv[3] : ".basis" );

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	pyro_vtree L(data, num_spinon_pairs);
	
	printf("Building state tree...\n");
	L.build_state_tree();
	
	FILE* outfile = std::fopen(outfilename.c_str(), "w");
	L.write_basis_file(outfile);
	std::fclose(outfile);

	return 0;
}
