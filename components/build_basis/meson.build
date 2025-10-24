
basis_inc = [include_directories('include'), vendor_inc, common_inc]

basis_common_src= [
  'src/pyro_tree.cpp',
  ]

spinon_basis_sources = files(['src/gen_spinon_basis.cpp'] + basis_common_src) 
spinon_basis_big_sources = files(['src/sbsearch.cpp'] + basis_common_src) 

partition_basis = executable('partition_basis', files('src/partition_basis.cpp'),
    dependencies: common_dep,
    include_directories: basis_inc,
    cpp_args: target_args
    )

polyring_basis = executable('polyring_basis', files('src/polyring_basis.cpp'), 
  dependencies: [common_dep, eigen_dep, operator_dep],
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )

gen_spinon_basis = executable('gen_spinon_basis',spinon_basis_sources, 
  dependencies: common_dep,
  include_directories: basis_inc,
  cpp_args: target_args,
  install: true
 )

sbsearch = executable('sbsearch', spinon_basis_big_sources, 
  dependencies: common_dep,
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )

merge_shards = executable('merge_shards', 'src/merge_shards.cpp',
  dependencies: common_dep,
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )

