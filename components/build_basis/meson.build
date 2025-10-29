
basis_inc = [include_directories('include'), vendor_inc, common_inc]

basis_common_src= [
  'src/pyro_tree.cpp',
  ]


partition_basis = executable('partition_basis', files('partition_basis.cpp'),
    dependencies: common_dep,
    include_directories: basis_inc,
    cpp_args: target_args
    )

polyring_basis = executable('polyring_basis', files('polyring_basis.cpp'), 
  dependencies: [common_dep, eigen_dep, operator_dep],
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )

gen_spinon_basis = executable('gen_spinon_basis',
  files(['gen_spinon_basis.cpp'] + basis_common_src), 
  dependencies: common_dep,
  include_directories: basis_inc,
  cpp_args: target_args,
  install: true
 )

sbsearch = executable('sbsearch', files(['sbsearch.cpp'] + basis_common_src) 
, 
  dependencies: common_dep,
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )


sbsearch_mpi = executable('sbsearch_mpi', 
  files(['sbsearch_mpi.cpp', 'src/pyro_tree_mpi.cpp'] + basis_common_src), 
  dependencies: [common_dep, mpi_dep],
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )

merge_shards = executable('merge_shards', 'merge_shards.cpp',
  dependencies: common_dep,
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )

