# Performance benchmarks (enabled with -Dbuild_benchmarks=true)

bench_bindir = join_paths(get_option('bindir'), 'bench')

benchmark_uint128 = executable('benchmark_uint128',
  'benchmark_uint128.cpp',
  include_directories: common_inc,
  cpp_args: target_args,
  install: false)

bench_apply = executable('bench_apply',
  'bench_apply.cpp',
  include_directories: [inc],
  dependencies: [operator_dep, linalg_routines_dep, common_dep],
  link_with: expectation_eval,
  cpp_args: target_args,
  install: true,
  install_dir: bench_bindir
  )

bench_permutations = executable('bench_permutations',
  'bench_permutations.cpp',
  include_directories: [inc],
  dependencies: [common_dep],
  cpp_args: target_args,
  install: true,
  install_dir: bench_bindir
  )

# Links bench_operator_mpi_lib (operator_mpi built with
# -DSUBSPACE_ED_BENCHMARK_OPERATIONS) for per-phase timer output.
bench_apply_mpi = executable('bench_apply_mpi',
  'bench_apply_mpi.cpp',
  include_directories: [inc, operator_inc],
  dependencies: [common_dep, eigen_dep, linalg_routines_dep, mpi_dep],
  link_with: [bench_operator_mpi_lib],
  cpp_args: target_args,
  install: true,
  install_dir: bench_bindir
  )

bench_lanczos = executable('bench_lanczos',
  'bench_lanczos.cpp',
  include_directories: [inc],
  dependencies: [operator_dep, linalg_routines_dep, common_dep],
  link_with: expectation_eval,
  cpp_args: target_args,
  install: true,
  install_dir: bench_bindir
  )
