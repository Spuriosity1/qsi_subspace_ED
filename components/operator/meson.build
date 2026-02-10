operator_inc = [include_directories('include'), common_inc]

operator_src = ['src/operator.cpp', 
#  'src/group_theory.cpp', 
  'src/operator_matrix.cpp']

operator_dep_set = [g_deps, eigen_dep, omp_dep, absl_dep ]

operator_lib = static_library('operator', operator_src,
  dependencies: operator_dep_set,
  include_directories: operator_inc,
  cpp_args: target_args
  )

operator_mpi_src = ['src/operator.cpp', 'src/operator_mpi.cpp']

operator_mpi_lib = static_library('operator_mpi', operator_mpi_src,
  dependencies: [g_deps, eigen_dep, mpi_dep, absl_dep ],
  include_directories: operator_inc,
  cpp_args: target_args
  )

# special bench versions
if get_option('build_benchmarks')
  bench_operator_mpi_lib = static_library('expectation_eval_mpi', operator_mpi_src,
    dependencies: [g_deps, eigen_dep, omp_dep, spectra_dep, mpi_dep, absl_dep], #todo purge spectra
    include_directories: operator_inc,
    link_with: operator_lib,
    cpp_args: [target_args, '-DSUBSPACE_ED_BENCHMARK_OPERATIONS']
    )
endif

# dependency declarations
operator_dep = declare_dependency(
  link_with: operator_lib,
  include_directories: operator_inc,
  dependencies: operator_dep_set
  )

operator_mpi_dep = declare_dependency(
  link_with: operator_mpi_lib,
  include_directories: operator_inc,
  dependencies: [hdf5_dep, mpi_dep]
  )
