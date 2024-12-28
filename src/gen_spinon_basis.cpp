#include "bittools.hpp"
#include "pyro_tree.hpp"
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>





using namespace std;
using json=nlohmann::json;

void print_tetra(const spin_set& t){
		printf("tetra at %p members [", static_cast<const void*>(&t) );
		for (auto si : t.member_spin_ids){printf( "%d, ", si);}
		printf("] bitmask ");
		auto b = t.bitmask;
		printf("0x%016llx%016llx\n", b.uint64[1],b.uint64[0]);
}

void print_spin(int idx, const spin& s){
	printf("Spin %d at %p ", idx, static_cast<const void*>(&s));
	printf("Neighbours:\n");
	for (auto t : s.tetra_neighbours){
		printf("\t");
		print_tetra(*t);
	}
}

int main (int argc, char *argv[]) {
	if (argc < 2) {
		printf("USAGE: %s <latfile: json\n", argv[0]);
	}

	std::string infilename(argv[1]);
	string outfilename=infilename.substr(0,infilename.find_last_of('.'))+".csv";

	ifstream ifs(infilename);
	json data = json::parse(ifs);
	ifs.close();

	pyro_tree L(data);

	printf("Building state tree...");
	L.build_state_tree();
	
#if VERBOSITY > 3
	for (auto t: L.tetras){	print_tetra(t); }
#endif
#if VERBOSITY > 2
	for (int si=0; si<L.spins.size(); si++){
		print_spin(si, L.spins[si]);
	}
	L.print_state_tree();
#endif
	

	FILE* outfile = std::fopen(outfilename.c_str(), "w");
	for (Uint128 b : L.get_states()){
		std::fprintf(outfile, "0x%016llx%016llx\n", b.uint64[1],b.uint64[0]);
	}
	std::fclose(outfile);

	return 0;
}
