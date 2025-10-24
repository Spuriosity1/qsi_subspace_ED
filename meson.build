project('projected_ed', ['cpp', 'c'],
  version: '1.0.0',
  meson_version : '>= 1.7.0',
  default_options : [ 'warning_level=1', 'buildtype=release', 'cpp_std=c++2a' ]
)

cmake = import('cmake')

# add_project_arguments(['
add_project_link_arguments(['-pthread', '-lstdc++'], language: 'cpp' )

# Compiler flags for 128-bit integer support
cpp_compiler = meson.get_compiler('cpp')
host_cpu = host_machine.cpu_family()

target_args = []

if get_option('buildtype') == 'release'
  add_project_arguments('-DNDEBUG', language: ['c', 'cpp'])
  if host_cpu.startswith('x86')
    if cpp_compiler.has_argument('-mavx2')
      target_args += ['-mavx2']
    elif cpp_compiler.has_argument('-mavx')
      target_args += ['-mavx']
    endif
    if cpp_compiler.has_argument('-msse2')
      target_args += ['-msse2']
    endif
  elif host_cpu.startswith('aarch64') or host_cpu.startswith('arm')
    #    if cpp_compiler.has_argument('-mfpu=neon')
    #      target_args += ['-mfpu=neon']
    #    elif cpp_compiler.has_argument('-march=armv8-a+simd')
    #      target_args += ['-march=armv8-a+simd']
    #    endif
    #    if cpp_compiler.has_argument('-march=armv8-a+sha3')
    #      target_args += ['-march=armv8-a+sha3']
    #    endif
  endif

  target_args += ['-DNDEBUG']

elif get_option('buildtype') == 'debug'
  # add_project_arguments(['-fsanitize=address'], language: 'cpp' )
endif



hdf5_dep = dependency('hdf5',
  language: 'c',
  required: false,
  not_found_message: 'dependency HDF5 not found')

json_dep = dependency('nlohmann_json', required: true)

eigen_dep = dependency('Eigen3', method: 'cmake',  modules: ['Eigen3::Eigen'], required: true).as_system()

spectra_dep = dependency('spectra', method: 'cmake')

omp_dep = dependency('OpenMP', required: true)
mpi_dep = dependency('mpi', language: 'cpp')


# MRRR parallel tridiagonal matrix eigensolver
mr3smp_proj = subproject('mr3smp')
mr3smp_dep = mr3smp_proj.get_variable('mr3smp_dep')


cmake_opts = cmake.subproject_options()
cmake_opts.add_cmake_defines({
  'CMAKE_SOURCE_DIR': '/dummy/path/not/equal',
  'ENABLE_WERROR': 'OFF'
})


# Local include dir for headers like bittools.hpp, admin.hpp, basis_io
inc = [ include_directories('include') ]

g_deps = [
  hdf5_dep,
  json_dep,
  omp_dep,
  eigen_dep,
  mr3smp_dep
]


subdir('src')

if eigen_dep.found()
if spectra_dep.found()

expectation_eval = static_library('expectation_eval', ['src/expectation_eval.cpp', 'src/operator_matrix.cpp', 'src/operator.cpp'],
  dependencies: [g_deps, eigen_dep, omp_dep, spectra_dep],
  include_directories: inc,
  link_with: operator_lib,
  cpp_args: target_args
  )

expectation_eval_mpi = static_library('expectation_eval_mpi', ['src/expectation_eval.cpp', 'src/operator_mpi.cpp', 'src/operator.cpp'],
  dependencies: [g_deps, eigen_dep, omp_dep, spectra_dep, mpi_dep], #todo purge spectra
  include_directories: inc,
  link_with: operator_lib,
  cpp_args: target_args
  )

diag_DOQSI_ham = executable('diag_DOQSI_ham',
  files(['src/diag_DOQSI_ham.cpp']),
  dependencies: [g_deps, eigen_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )

build_hamiltonian = executable('build_hamiltonian',
  files(['src/build_hamiltonian.cpp', 'src/expectation_eval.cpp']),
  dependencies: [g_deps, eigen_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )

#ftlm_DOQSI_ham = executable('ftlm_DOQSI_ham',
#  files(['src/ftlm_DOQSI_ham.cpp', 'src/expectation_eval.cpp','src/ftlm.cpp']),
#  dependencies: [g_deps, eigen_dep, spectra_dep],
#  include_directories: inc,
#  cpp_args: target_args
#  )


eval_observables = executable('eval_observables',
  files(['src/eval_observables.cpp', 'src/expectation_eval.cpp']),
  dependencies: [g_deps, eigen_dep, omp_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )


eval_dsf = executable('eval_dsf',
  files(['src/eval_dsf.cpp', 'src/expectation_eval.cpp']),
  dependencies: [g_deps, eigen_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )


endif 
endif


subdir('bench')
subdir('test')
