# umap_memtest = executable('umap_memtest', 'src/umap_memtest.cpp', include_directories: inc)
# umap_memtest = executable('map_memtest', 'src/map_memtest.cpp', include_directories: inc)
# umap_memtest = executable('vector_memtest', 'src/vector_memtest.cpp', include_directories: inc)
# 
# 
# udense_dep = dependency('unordered_dense', required:false)
# 
# if udense_dep.found()
#   umap_memtest = executable('ankerl_memtest', 'src/ankerl_memtest.cpp', include_directories: inc, dependencies: udense_dep)
#   map_choice = executable('map_choice', 'src/map_choice.cpp', include_directories: inc, dependencies: udense_dep)
# endif




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
  
