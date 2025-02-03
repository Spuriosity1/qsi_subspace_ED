# QSI subspace ED

## Basic description of the algorithm
The C++ part is designed to efficiently enumerate ice states. 

Input: 
- A list of atoms $A$
- A set of hexagons $h$ 
- A set of tetrahedron-constraints $t$, which are lists of four atom indices.

The algorithm represents basis states (in the $Z$ basis, say) by 128-bit binary strings, with 0 as 'down' and 1 as up.

A given 'node' in the tree has the form
```
struct vtree_node_t {
	Uint128 state_thus_far;
	unsigned curr_spin;
	// curr_spin is the bit ID of the rightmost unknown spin
	// i.e. (1<<curr_spin) & state_thus_far is guaranteed to be 0
};
```

The main algorithm is depth-first search. For this, we generate a stack of nodes `to_examine`, adding compatible states to the stack on each iteration. DFS is appropriate here because the maximum depth is known (equal to the number of spins), while the maximum breadth is very large. DFS restricts the stack size to easily fit in memory.

For completeness, BFS was also implemented in order to generate a set of balanced starting points for the parallel algorithm. 


## Obtaining and Compiling
From the root directory:
```bash
git clone https://github.com/Spuriosity1/qsi_subspace_ED.git
cd qsi_subspace_ED
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

