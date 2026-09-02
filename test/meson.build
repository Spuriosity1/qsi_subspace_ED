# Shell-driven integration tests
spinon_tests = find_program('test_spinonfree_222.sh')
test('gen_spinon_basis_st', spinon_tests, args: ['single'])
test('gen_spinon_basis_par', spinon_tests, args: ['parallel'])
test('gen_spinon_basis_spinorder', spinon_tests, args: ['spinorder'])

sbsearch_test = find_program('check_sbsearch_against_known.sh')
test('check_sbsearch_against_known', sbsearch_test)


# Unit test executables
test_uint128 = executable('test_uint128',
  'test_uint128.cpp',
  dependencies: common_dep,
  install: false)

test_PMR_operator = executable('test_PMR_operator',
  'test_PMR_construct.cpp',
  include_directories: inc,
  dependencies: operator_dep,
  install: false
  )

test_permutations = executable('test_permutations',
  'test_permutations.cpp',
  include_directories: inc,
  dependencies: common_dep,
  cpp_args: target_args,
  install: false
  )

test_lanczos = executable('test_lanczos',
  'test_lanczos.cpp',
  include_directories: inc,
  dependencies: [operator_mpi_dep, eigen_dep, linalg_routines_dep],
  install: false
  )

# Manual-run executable: needs a lattice file argument
test_apply = executable('test_apply',
  'test_apply.cpp',
  include_directories: inc,
  dependencies: [operator_dep, linalg_routines_dep, common_dep],
  install: false
  )

test_num_nonvanishing = executable('test_num_nonvanishing', 
  'test_num_nonvanishing.cpp', 
  include_directories: inc,
  dependencies: [operator_dep, linalg_routines_dep, common_dep],
  install: false
  )

test_apply_mpi = executable('test_apply_mpi',
  'test_apply_mpi.cpp',
  include_directories: inc,
  dependencies: [operator_mpi_dep, common_dep, linalg_routines_dep, operator_dep],
  install: false
  )

test_mpi_alignment = executable('test_mpi_alignment',
  ['test_mpi_alignment.cpp', basis_common_mpi_src],
  include_directories: inc + basis_inc,
  dependencies: [mpi_dep, common_dep],
  install: false
  )


# Test registration
test('PMRoperator_edge_case', test_PMR_operator,
  timeout: 10,
  suite: ['unit', 'constructors']
  )

test('uint128_constructors', test_uint128,
     args: ['--test-constructors'],
     timeout: 10,
     suite: ['unit', 'constructors'])

test('uint128_operators', test_uint128,
     args: ['--test-operators'],
     timeout: 10,
     suite: ['unit', 'operators'])

test('uint128_bit_operations', test_uint128,
     args: ['--test-bit-ops'],
     timeout: 10,
     suite: ['unit', 'bit_operations'])

test('uint128_hash_equality', test_uint128,
     args: ['--test-hash'],
     timeout: 10,
     suite: ['unit', 'hash'])

test('uint128_edge_cases', test_uint128,
     args: ['--test-edge-cases'],
     timeout: 10,
     suite: ['unit', 'edge_cases'])

test('test_permutations', test_permutations,
  timeout: 30,
  suite: ['unit', 'permute']
  )

test('test_lanczos', test_lanczos, timeout: 180)

fs = import('fs')
mpiexec = find_program('mpiexec', 'mpirun', required: false)
if mpiexec.found()
  test('test_mpi_alignment',
    mpiexec,
    args: ['-n', '2', test_mpi_alignment.full_path()],
    timeout: 60,
    suite: ['unit', 'mpi'],
    is_parallel: false,
  )

  # Local variant: uses the small in-repo basis, exercises the pipelined
  # off-diagonal apply across the 3 MPI basis search structures (BST, fast,
  # interp) against the serial reference, on 2 ranks.
  local_lattice_dir = meson.current_source_dir() / 'lattice_files'
  test('test_apply_mpi_local',
    mpiexec,
    args: [
      '-n', '2',
      test_apply_mpi.full_path(),
      local_lattice_dir / 'pyro_2_2_3x0,4,4b4,0,4b4,4,0b1.json',
      '-b', local_lattice_dir / 'pyro_2_2_3x0,4,4b4,0,4b4,4,0b1.0test.h5',
    ],
    timeout: 300,
    suite: ['mpi'],
    is_parallel: false,
  )

  # Needs a pre-generated basis next to the lattice file; only available on
  # machines with the external lattice_files directory (e.g. the HPC).
  lattice_files_dir = meson.project_source_root() / '../lattice_files'
  if fs.exists(lattice_files_dir / 'pyro_2,0,0_0,3,0_0,0,3.json')
    test('test_apply_mpi',
      mpiexec,
      args: [
        '-n', '4',
        test_apply_mpi.full_path(),
        lattice_files_dir / 'pyro_2,0,0_0,3,0_0,0,3.json',
        '--sector', 'basis_s8.10.9.9',
      ],
      timeout: 600,
      is_parallel: false,
    )
  endif
endif
