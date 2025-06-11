#include <iostream>
#include <cstdint>
#include <ankerl/unordered_dense.h>
#include "bittools.hpp"

using namespace std;

int main (int argc, char *argv[]) {
	if (argc < 2){
		cout << "Usage: "<<argv[0]<<" <N>"<<std::endl;
		return 1;
	}	

	uint64_t N = atoi(argv[1]);

	ankerl::unordered_dense::map<Uint128, size_t, Uint128Hash, Uint128Eq> m;
	for (uint64_t i=0; i<N; ++i){
		m[i] = i;
	}
	return 0;
}

