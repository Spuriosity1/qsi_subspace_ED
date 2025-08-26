# QSI subspace ED

This is a low level template library + set of executables designed to do exact diagonalisaiton within subspaces of spin Hamiltonians. This project was originally motivated by quantum spin ice, however there may be other applications.

Executables:
```
build_hamiltonian
diag_DOQSI_ham
eval_dsf
eval_observables
gen_projected_basis
gen_spinon_basis
partition_basis
```

### `build_hamiltonian` 
```
Usage: build_ham [--help] [--version] [--sector VAR] [[--B VAR...]|[--g VAR...]] [--Jpm VAR] --output_dir VAR [--n_spinons VAR] lattice_file

Positional arguments:
  lattice_file      

Optional arguments:
  -h, --help        shows help message and exits 
  -v, --version     prints version information and exits 
  -s, --sector      
  --B               magnetic field, units of Jzz [nargs: 3] 
  --g               raw ring exchange, units of Jzz [nargs: 4] 
  --Jpm             Jom, units of Jzz 
  -o, --output_dir  output directory [required]
  --n_spinons       [nargs=0..1] [default: 0]
```




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

### Dependencies
- HDF5
- nlohmann::json.

From the root directory:
```bash
git clone https://github.com/Spuriosity1/qsi_subspace_ED.git
cd qsi_subspace_ED
meson setup build
ninja -C build
ninja -C build test # optional
```


