
common_inc = include_directories('include')

common_src = ['src/tetra_graph_io.cpp']

common_lib = static_library('common', common_src,
  dependencies: json_dep,
  include_directories: common_inc,
  cpp_args: target_args
  )

common_dep = declare_dependency(
  link_with: common_lib,
  include_directories: common_inc,
  dependencies: [hdf5_dep, json_dep]
  )
