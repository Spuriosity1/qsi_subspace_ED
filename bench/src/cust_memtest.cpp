#include <iostream>
#include <cstdint>
#include <map>
#include "bittools.hpp"

using namespace std;

using idx_t = uint64_t;
using state_t = UInt128;


int main (int argc, char *argv[]) {
	if (argc < 2){
		cout << "Usage: "<<argv[0]<<" <N>"<<std::endl;
		return 1;
	}	

	uint64_t N = atoi(argv[1]);

    BasisIndex b;
	for (uint64_t i=0; i<N; ++i){
		m[i] = i;
	}
	return 0;
}

