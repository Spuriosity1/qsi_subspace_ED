#include <bitset>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <ostream>

int main (int argc, char *argv[]) {
	using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

	__int128 mask = 0;
	std::bitset<128> bs_mask;

	int N = atoi(argv[1]);

	auto tic0= high_resolution_clock::now();
	for (int i=0; i<N; i++){
	}
	auto toc0 = high_resolution_clock::now();
	
	auto tic1= high_resolution_clock::now();
	for (int i=0; i<N; i++){
		mask ^= (static_cast<__int128>(1) << i%128);
		mask == static_cast<__int128>(0xffffff);
	}
	auto toc1 = high_resolution_clock::now();
	auto tic2= high_resolution_clock::now();
	for (int i=0; i<N; i++){
		bs_mask ^= (static_cast<__int128>(1) << i%128);
		bs_mask == static_cast<__int128>(0xffffff);
	}
	auto toc2 = high_resolution_clock::now();


	std::cout<<"Calibration time: "<<duration_cast<milliseconds>(toc0-tic0).count()*1./N<<"ms\n";
	std::cout<<"__int128 time: "<<duration_cast<milliseconds>(toc1-tic1).count()*1./N<<"ms\n";
	std::cout<<"bitmask time: "<<duration_cast<milliseconds>(toc2-tic2).count()*1./N<<"ms\n";

	return 0;
}
