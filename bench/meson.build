
# Benchmark test (if you want performance testing)
benchmark_uint128 = executable('benchmark_uint128',
                              'benchmark_uint128.cpp',
                              include_directories: common_inc,
                              install : false)

bench_apply = executable('bench_apply',
  'bench_apply.cpp',
  include_directories: [inc],
  dependencies: [operator_dep, linalg_routines_dep, common_dep],
  link_with: expectation_eval,
  install: false
  )


bench_permutations = executable('bench_permutations',
  'bench_permutations.cpp',
  include_directories: [inc],
  dependencies: [common_dep],
  cpp_args: target_args,
  install: false
  )


bench_apply_mpi = executable('bench_apply_mpi',
  'bench_apply_mpi.cpp',
  include_directories: [inc, operator_inc],
  dependencies: [common_dep, eigen_dep,  linalg_routines_dep, mpi_dep],
  link_with: [bench_operator_mpi_lib],
  install: false
  )

bench_apply_mpi = executable('bench_apply_mpi_omp',
  'bench_apply_mpi.cpp',
  include_directories: [inc, operator_inc],
  dependencies: [common_dep, eigen_dep, linalg_routines_dep, mpi_dep, omp_dep],
  link_with: [bench_operator_mpi_lib],
  install: false
  )
  

bench_lanczos = executable('bench_lanczos',
  'bench_lanczos.cpp',
  include_directories: [inc],
  dependencies: [operator_dep, linalg_routines_dep, common_dep],
  link_with: expectation_eval,
  install: false
  )

bench_lanczos_mpi = executable('bench_lanczos_mpi',
  'bench_lanczos_mpi.cpp',
  include_directories: [inc, operator_inc],
  dependencies: [common_dep, eigen_dep, linalg_routines_mpi_dep],
  link_with: [operator_mpi_lib],
  install: false
  )
  
