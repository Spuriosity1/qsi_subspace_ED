
linalg_routines_inc = [include_directories('include'), common_inc]
linalg_routines_src = ['src/lanczos.cpp']


linalg_routines_lib = static_library('linalg_routines', linalg_routines_src, 
  dependencies: [ lapack_dep, eigen_dep, blas_dep],
  include_directories: linalg_routines_inc,
  cpp_args: target_args
)



linalg_routines_dep = declare_dependency(
  include_directories: linalg_routines_inc,
  link_with: linalg_routines_lib,
  dependencies: [ lapack_dep, blas_dep]
)

