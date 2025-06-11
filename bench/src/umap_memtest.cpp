#include <iostream>
#include <unordered_map>
#include <cstdint>
#include <map>
#include "bittools.hpp"

using namespace std;

// Hash function for Uint128 to use in unordered_map
struct Uint128Hash {
	std::size_t operator()(const Uint128& b) const {
		return std::hash<uint64_t>()(b.uint64[0]) ^ std::hash<uint64_t>()(b.uint64[1]);
	}
};

struct Uint128Eq {
	bool operator()(const Uint128& a, const Uint128& b) const {
		return (a.uint64[0] == b.uint64[0]) && (a.uint64[1] == b.uint64[1]);
	}
};


int main (int argc, char *argv[]) {
	if (argc < 2){
		cout << "Usage: "<<argv[0]<<" <N>"<<std::endl;
		return 1;
	}	

	uint64_t N = atoi(argv[1]);

	std::unordered_map<Uint128, int, Uint128Hash, Uint128Eq> m;

	for (uint64_t i=0; i<N; ++i){
		m[i] = i;
	}
	return 0;
}

