#include "basis_io.hpp"
#include "pyro_tree.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include "admin.hpp"
#include "bittools.hpp"


using namespace std;
using json=nlohmann::json;


// Checks that round-trip basis save/load mathces
//
int main (int argc, char *argv[]) {
	if (argc < 2) {
		printf("USAGE: %s <latfile: json>\n", argv[0]);
		return 1;
	}

	std::string infilename(argv[1]);

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	unsigned num_spinon_pairs=0;

	basis_tree_builder L(data, num_spinon_pairs);
	
	printf("Building state tree...\n");
	L.build_state_tree();

	printf("Sorting...\n");
	L.sort();


	auto outfilename=as_basis_file(infilename, ".0.basis" );
	
	L.write_basis_csv(outfilename);
	L.write_basis_hdf5(outfilename);

	auto v2 = basis_io::read_basis_csv(outfilename);
	auto v1 = basis_io::read_basis_hdf5(outfilename);

	assert( v1.size() == v2.size() );
	size_t n_failures = 0;

	for (size_t i=0; i<v1.size(); i++){
		if (!(v1[i] == v2[i])) {
			std::cerr << "Discrepancy on line " << i << ": " << v1[i]
			   << " != " << v2[i] << std::endl;
		  n_failures++;
		}
	}
	cout<<"Test complete, "<<n_failures<<" of "<<v1.size()<<" rows disagree"<<std::endl;

	return n_failures;	

}

