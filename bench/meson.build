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
                              include_directories: inc,
                              install : false)


bench_operator_mpi_lib = static_library('expectation_eval_mpi', ['../src/operator_mpi.cpp', '../src/operator.cpp'],
  dependencies: [g_deps, eigen_dep, omp_dep, spectra_dep, mpi_dep], #todo purge spectra
  include_directories: inc,
  link_with: operator_lib,
  cpp_args: [target_args, '-DSUBSPACE_ED_BENCHMARK_OPERATIONS']
  )


bench_apply = executable('bench_apply',
  'bench_apply.cpp',
  include_directories: inc,
  dependencies: [g_deps, eigen_dep, spectra_dep],
  link_with: expectation_eval,
  install: false
  )


bench_apply_mpi = executable('bench_apply_mpi',
  'bench_apply_mpi.cpp',
  include_directories: inc,
  dependencies: [g_deps, eigen_dep, spectra_dep, mpi_dep],
  link_with: [bench_operator_mpi_lib],
  install: false
  )
  

# Register benchmark (optional)
benchmark('uint128_performance', benchmark_uint128,
         timeout : 60,
         suite : 'benchmark')
