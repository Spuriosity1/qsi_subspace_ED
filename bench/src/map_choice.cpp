#include <iostream>
#include <cstdint>
#include <map>
#include <ankerl/unordered_dense.h>
#include "bittools.hpp"

using namespace std;

// For benchmarking construction and access time

int main (int argc, char *argv[]) {
	if (argc < 2){
		cout << "Usage: "<<argv[0]<<" <N>"<<std::endl;
		return 1;
	}	

	uint64_t N = atoi(argv[1]);

	cout << "Constructing...\n";

	std::map<Uint128, size_t> map;
	for (uint64_t i=0; i<N; ++i){
		map[i] = i;
	}

	std::unordered_map<Uint128, size_t, Uint128Hash, Uint128Eq> umap;
	return 0;
}

