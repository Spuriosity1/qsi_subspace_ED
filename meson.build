project('projected_ed', ['cpp', 'c'],
  version: '1.0.0',
  meson_version : '>= 1.3.0',
  default_options : [ 'warning_level=3', 'buildtype=release', 'cpp_std=c++2a' ]
)

if get_option('buildtype') == 'release'
  add_project_arguments('-Ofast', language : 'cpp')
endif

hdf5_dep = dependency('hdf5',
  language: 'c',
  required: false,
  not_found_message: 'dependency HDF5 not found')

json_dep = dependency('nlohmann_json', required: true)

eigen_dep = dependency('eigen3', required: true)

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


subdir('bench')
