
build_hamiltonian_sources = files(['build_hamiltonian.cpp'])

tetra_io_lib = static_library('tetra_graph_io_lib', ['tetra_graph_io.cpp'],
  include_directories: inc,
  cpp_args: target_args,
  dependencies: g_deps
  )

operator_lib = static_library('operator_lib', ['operator.cpp'],
  dependencies: [g_deps, eigen_dep, omp_dep ],
  include_directories: inc,
  cpp_args: target_args
  )

subdir('basis')

