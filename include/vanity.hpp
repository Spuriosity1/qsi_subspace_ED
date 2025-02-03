#pragma once
#include "pyro_tree.hpp"


inline void print_tetra(const spin_set& t){
		printf("tetra at %p members [", static_cast<const void*>(&t) );
		for (auto si : t.member_spin_ids){printf( "%d, ", si);}
		printf("] bitmask ");
		auto b = t.bitmask;
		printf("0x%016llx%016llx\n", b.uint64[1],b.uint64[0]);
}

inline void print_spin(int idx, const spin& s){
	printf("Spin %d at %p ", idx, static_cast<const void*>(&s));
	printf("Neighbours:\n");
	for (auto t : s.tetra_neighbours){
		printf("\t");
		print_tetra(*t);
	}
}
