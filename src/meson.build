# Application executables.
# Paths are relative to this directory; deps/includes come from the root
# and components/ meson files (subdir'd before src/).

polyring_basis = executable('polyring_basis', files('polyring_basis.cpp'),
  dependencies: [common_dep, eigen_dep, operator_dep],
  include_directories: basis_inc,
  cpp_args: target_args,
  install: true
  )

polyring_basis_mpi = executable('polyring_basis_mpi', files('polyring_basis_mpi.cpp'),
  dependencies: [common_dep, eigen_dep, operator_dep, mpi_dep],
  include_directories: basis_inc,
  cpp_args: target_args,
  install: true
  )

expectation_eval = static_library('expectation_eval', 'expectation_eval.cpp',
  include_directories: inc,
  dependencies: [operator_dep, linalg_routines_dep, g_deps],
  cpp_args: target_args)

diag_DOQSI_ham = executable('diag_DOQSI_ham',
  files(['diag_DOQSI_ham.cpp']),
  dependencies: [g_deps, eigen_dep, operator_dep, linalg_routines_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args,
  install: true
  )

diag_DOQSI_ham_mpi = executable('diag_DOQSI_ham_mpi',
  files(['diag_DOQSI_ham_mpi.cpp']),
  dependencies: [g_deps, eigen_dep, operator_mpi_dep, linalg_routines_mpi_dep],
  include_directories: inc,
  link_with: [expectation_eval],
  cpp_args: target_args,
  install: true
  )

eval_observables = executable('eval_observables',
  files(['eval_observables.cpp']),
  dependencies: [g_deps, eigen_dep, omp_dep, operator_dep, linalg_routines_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )

ring_normed = executable('ring_normed',
  files(['ring_normed.cpp']),
  dependencies: [g_deps, eigen_dep, omp_dep, operator_dep, linalg_routines_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )

eval_observables_mpi = executable('eval_observables_mpi',
  files(['eval_observables_mpi.cpp']),
  dependencies: [g_deps, eigen_dep, omp_dep, operator_mpi_dep, linalg_routines_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )
