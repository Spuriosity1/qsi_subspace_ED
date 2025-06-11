#include <iostream>
#include <cstdint>
#include <vector>
#include "bittools.hpp"

using namespace std;

int main (int argc, char *argv[]) {
	if (argc < 2){
		cout << "Usage: "<<argv[0]<<" <N>"<<std::endl;
		return 1;
	}	

	uint64_t N = atoi(argv[1]);

	std::vector<Uint128> m;
    m.resize(N);
	for (uint64_t i=0; i<N; ++i){
		m[i] = i;
	}
	return 0;
}

