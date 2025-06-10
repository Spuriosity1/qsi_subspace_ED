project('projected_ed', ['cpp', 'c'],
  version: '1.0.0',
  meson_version : '>= 1.3.0',
  default_options : [ 'warning_level=3', 'buildtype=release', 'cpp_std=c++2a' ]
)

if get_option('buildtype') == 'release'
  add_project_arguments('-Ofast',  language : 'cpp')
endif


hdf5_dep = dependency('hdf5',
                      language: 'c',
                      required: false,
                      not_found_message: 'dependency HDF5 not found')

json_dep = dependency('nlohmann_json', required: true)

inc = [
  include_directories('include')
  ]

g_deps = [
  hdf5_dep,
  json_dep
  ]

subdir('src')

