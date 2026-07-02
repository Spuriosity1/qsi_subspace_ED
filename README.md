# QSI subspace ED

This is a collection of C++ programs designed for work in subspaces of locally-constrained spin Hamiltonians.

## Obtaining and Compiling

### Dependencies
- HDF5
- nlohmann::json.
- MPI

From the root directory:
```bash
git clone https://github.com/Spuriosity1/qsi_subspace_ED.git
cd qsi_subspace_ED
meson setup build --prefix `pwd` # or your choice of install location
ninja -C build install
meson test -C build # optional
```

Note: the `test_apply_mpi` integration test is only registered when the
external `../lattice_files` directory (with pre-generated bases) exists next
to the repository, as on the HPC.

## Repository layout

```
components/
  common/           Uint128 bit tools, basis HDF5/CSV IO, lattice graph
                    parsing (tetra_graph_io), MPI context + state->rank hash
  build_basis/      ice-state enumeration (DFS over constraint tree, serial
                    and MPI), shard sorting/merging, basis partitioning
  operator/         symbolic PMR operators, basis search structures
                    (ZBasisBST / ZBasisInterp / ZBasisBSTFast), matrix-free
                    apply: LazyOpSum (operator_matrix) and MPILazyOpSum
                    (operator_mpi: pipeline + batched alltoallv paths)
  linalg_routines/  Lanczos (serial and MPI), BLAS/LAPACK adapters
include/            app-level glue headers (hamiltonian_setup, CLI bits,
                    physics/ ring-exchange definitions)
src/                application executables (meson targets live here)
test/               unit + integration tests (meson test)
bench/              performance benchmarks (-Dbuild_benchmarks=true)
scripts/            python drivers, plotting, lattice-file generation
driver/, hpc_driver/  batch/SLURM job scripts
jl/                 Julia post-processing (FTLM thermodynamics)
vendor/, subprojects/ third-party headers and meson wraps
```

### Benchmarking
The bash script bench/benchme.sh (run it from the project root directory) 
runs a number of calculations on unit cells 1x2xj, for j=1:10. Runtimes
are spat out as raw numbers.

```bash
bench/benchme.sh 4
0.01
```

## Introduction

This is a low level template library + set of executables designed to do exact diagonalisaiton within subspaces of spin Hamiltonians. This project was originally motivated by quantum spin ice, however there may be other applications.

The pipeline is as follows:

![](https://raw.githubusercontent.com/Spuriosity1/qsi_subspace_ED/refs/heads/main/projected%20ED.drawio.svg)


Executables (basis generation): `gen_spinon_basis`, `sbsearch`, `sbsearch_mpi`,
`sort_shard`, `merge_shards`, `merge_sorted_shards`, `partition_shard`,
`partition_basis`, `polyring_basis`, `polyring_basis_mpi`.

Executables (diagonalisation / evaluation): `diag_DOQSI_ham`,
`diag_DOQSI_ham_mpi`, `eval_observables`, `eval_observables_mpi`,
`ring_normed`.



### `gen_spinon_basis`
```
Usage: gen_spinon_basis [--help] [--version] [--n_threads VAR] [--order_spins VAR] [--out_format VAR] lattice_file n_spinon_pairs extension

Positional arguments:
  lattice_file    The json-vlaued lattice spec 
  n_spinon_pairs  [nargs=0..1] [default: 0]
  extension       [nargs=0..1] [default: ".basis"]

Optional arguments:
  -h, --help      shows help message and exits 
  -v, --version   prints version information and exits 
  --n_threads     Number of threads to distribute across (works best as a power of 2)Setting to 0 will use the single-threaded implementations [nargs=0..1] [default: 0]
  --order_spins   [nargs=0..1] [default: "greedy"]
  --out_format    [nargs=0..1] [default: "h5"]
```
lattice.json required fields:
```json
{
    "tetrahedra": [
        {
            "member_spin_idx": [
                0,
                8,
                16,
                24
            ]
        },
        ...
    ]
}
```
The output is a HDF5 file in the Z basis of all states satisfying Q=0 (or Q=+/- 1/2 if odd membered). If 

Examine the output with `h5dump --contents ../lattice_files/pyro_2,0,0_0,2,0_0,0,2.0.basis.h5` (ensure that HDF5 is in your `PATH`)


### `diag_DOQSI_ham`
```
Usage: diag_DOQSI_ham [--help] [--version] [--sector VAR] [[--B VAR...]|[--g VAR...]] [--Jpm VAR] --output_dir VAR [--ncv VAR] [--n_eigvals VAR] [--n_eigvecs VAR] [--n_spinons VAR] [--save_matrix] [--max_iters VAR] [--tol VAR] [--algorithm VAR] lattice_file

Positional arguments:
  lattice_file      

Optional arguments:
  -h, --help        shows help message and exits 
  -v, --version     prints version information and exits 
  -s, --sector      a key to the basis file
  --B               magnetic field, units of Jzz [nargs: 3] 
  --g               raw ring exchange, units of Jzz [nargs: 4] 
  --Jpm             Jom, units of Jzz 
  -o, --output_dir  output directory [required]
  -k, --ncv         Krylov dimension, should be > 2*n_eigvals [nargs=0..1] [default: 15]
  -n, --n_eigvals   Number of eigenvlaues to compute [nargs=0..1] [default: 5]
  -N, --n_eigvecs   Number of eigenvectors to store (must be <= n_eigvals) [nargs=0..1] [default: 4]
  --n_spinons       [nargs=0..1] [default: 0]
  --save_matrix     Flag to get the solver to export a rep of the matrix 
  --max_iters       Max steps for iterative solver [nargs=0..1] [default: 1000]
  --tol             Tolerance iterative solver [nargs=0..1] [default: 1e-10]
  -a, --algorithm   Variant of ED algorithm to run. dense is best for small problems, mfsparse is a matrix free method that trades off speed for memory. 
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



