project('spectra', 'cpp',
  version: '0.9.0',
  meson_version: '>= 0.54.0'
)

eigen_dep = dependency('Eigen3',
  method: 'cmake',
  modules: ['Eigen3::Eigen'],
  required: true
).as_system()

spectra_dep = declare_dependency(
  include_directories: include_directories('include'),
  dependencies: eigen_dep
)
