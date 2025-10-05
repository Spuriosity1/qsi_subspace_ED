project('projected_ed', ['cpp', 'c'],
  version: '1.0.0',
  meson_version : '>= 1.3.0',
  default_options : [ 'warning_level=3', 'buildtype=release', 'cpp_std=c++2a' ]
)


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
    if cpp_compiler.has_argument('-mfpu=neon')
      target_args += ['-mfpu=neon']
    elif cpp_compiler.has_argument('-march=armv8-a+simd')
      target_args += ['-march=armv8-a+simd']
    endif
    if cpp_compiler.has_argument('-march=armv8-a+sha3')
      target_args += ['-march=armv8-a+sha3']
    endif
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

eigen_dep = dependency('Eigen3', method: 'cmake',  modules: ['Eigen3::Eigen'], required: true)

spectra_dep = dependency('spectra', method: 'cmake')

omp_dep = dependency('OpenMP', required: true)

#abseil_dep = dependency('absl', method: 'cmake', modules: ['absl::flat_hash_map'], required: true)
abseil_proj = subproject('abseil-cpp')
abseil_dep = abseil_proj.get_variable('absl_container_dep')

# MRRR parallel tridiagonal matrix eigensolver
mr3smp_proj = subproject('mr3smp')
mr3smp_dep = mr3smp_proj.get_variable('mr3smp_dep')

# Local include dir for headers like bittools.hpp, admin.hpp, basis_io
inc = [ include_directories('include') ]

g_deps = [
  hdf5_dep,
  json_dep,
  omp_dep,
  eigen_dep,
  mr3smp_dep,
  abseil_dep
]

subdir('src')

if hdf5_dep.found()
  
    partition_basis = executable('partition_basis', files('src/partition_basis.cpp'),
            dependencies: g_deps,
            include_directories:inc,
            cpp_args: target_args
            )


else
  add_project_arguments('-DDONT_USE_HDF5', language: 'cpp')
endif

gen_spinon_basis = executable('gen_spinon_basis',spinon_basis_sources, 
  dependencies: g_deps,
  include_directories:inc,
  cpp_args: target_args
 )


sbsearch = executable('sbsearch', spinon_basis_big_sources, 
  dependencies: g_deps,
  include_directories:inc,
  cpp_args: target_args
  )

merge_shards = executable('merge_shards', 'src/merge_shards.cpp',
  dependencies: g_deps,
  include_directories:inc,
  cpp_args: target_args
  )

if eigen_dep.found()

gen_projected_basis = executable('gen_projected_basis', files('src/gen_projected_basis.cpp'), 
  dependencies: [g_deps, eigen_dep, abseil_dep],
  include_directories:inc,
  cpp_args: target_args
  )

if spectra_dep.found()

expectation_eval = static_library('expectation_eval', ['src/expectation_eval.cpp', 'src/operator_matrix.cpp'],
  dependencies: [g_deps, eigen_dep, omp_dep, spectra_dep],
  include_directories: inc,
  cpp_args: target_args
  )

diag_DOQSI_ham = executable('diag_DOQSI_ham',
  files(['src/diag_DOQSI_ham.cpp']),
  dependencies: [g_deps, eigen_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )


#gs_DOQSI_ham = executable('gs_DOQSI_ham',
#  files(['src/gs_DOQSI_ham.cpp', 'src/expectation_eval.cpp']),
#  dependencies: [g_deps, eigen_dep, spectra_dep],
#  include_directories: inc,
#  cpp_args: target_args
#  )
#
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
