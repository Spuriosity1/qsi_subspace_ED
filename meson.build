project('projected_ed', ['cpp', 'c'],
  version: '1.0.0',
  meson_version : '>= 1.3.0',
  default_options : [ 'warning_level=3', 'buildtype=release', 'cpp_std=c++2a' ]
)


# Compiler flags for 128-bit integer support
cpp_compiler = meson.get_compiler('cpp')
host_cpu = host_machine.cpu_family()
add_project_arguments('-DUINT128_AVAILABLE', language : 'cpp')

if get_option('buildtype') == 'release'
  if host_cpu.startswith('x86')
    if cpp_compiler.has_argument('-mavx2')
      add_project_arguments('-mavx2', language : 'cpp')
    elif cpp_compiler.has_argument('-mavx')
      add_project_arguments('-mavx', language : 'cpp')
    elif cpp_compiler.has_argument('-msse2')
      add_project_arguments('-msse2', language : 'cpp')
    endif
  elif host_cpu.startswith('aarch64') or host_cpu.startswith('arm')
    if cpp_compiler.has_argument('-mfpu=neon')
      add_project_arguments('-mfpu=neon', language : 'cpp')
    elif cpp_compiler.has_argument('-march=armv8-a+simd')
      add_project_arguments('-march=armv8-a+simd', language : 'cpp')
    endif
    if cpp_compiler.has_argument('-march=armv8-a+sha3')
      add_project_arguments('-march=armv8-a+sha3', language : 'cpp')
    endif
  endif
endif

hdf5_dep = dependency('hdf5',
  language: 'c',
  required: false,
  not_found_message: 'dependency HDF5 not found')

json_dep = dependency('nlohmann_json', required: true)

eigen_dep = dependency('Eigen3', method: 'cmake',  modules: ['Eigen3::Eigen'])

spectra_dep = dependency('spectra', method: 'cmake', required: true)

# Local include dir for headers like bittools.hpp, admin.hpp, basis_io
inc = [ include_directories('include') ]

g_deps = [
  hdf5_dep,
  json_dep
]

subdir('src')

# Add diagonalize_rings
build_hamiltonian = executable('build_hamiltonian',
  build_hamiltonian_sources,
  dependencies: [g_deps, eigen_dep, spectra_dep],
  include_directories: inc)

gen_spinon_basis = executable('gen_spinon_basis', spinon_basis_sources, 
  dependencies: g_deps,
  include_directories:inc)

gen_spinon_basis_parallel = executable('gen_spinon_basis_parallel',spinon_basis_parallel_sources, 
  dependencies: g_deps,
  include_directories:inc)

gen_projected_basis = executable('gen_projected_basis', files('src/gen_projected_basis.cpp'), 
  dependencies: [g_deps, eigen_dep],
  include_directories:inc)


subdir('bench')
subdir('test')
