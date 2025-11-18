
basis_inc = [include_directories('include'), vendor_inc, common_inc]

basis_common_src= files( [
  'src/pyro_tree.cpp',
  ])

basis_common_mpi_src = files([
'src/pyro_tree_mpi.cpp',
  'src/pyro_tree.cpp'
])


partition_basis = executable('partition_basis', files('partition_basis.cpp'),
    dependencies: [common_dep, mpi_dep],
    include_directories: basis_inc,
    cpp_args: target_args
    )



gen_spinon_basis = executable('gen_spinon_basis',
  [files(['gen_spinon_basis.cpp']), basis_common_src], 
  dependencies: common_dep,
  include_directories: basis_inc,
  cpp_args: target_args,
  install: true
 )

sbsearch = executable('sbsearch', 
  [files(['sbsearch.cpp']), basis_common_src],
  dependencies: common_dep,
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )


sbsearch_mpi = executable('sbsearch_mpi', 
  [files(['sbsearch_mpi.cpp']),  basis_common_mpi_src],
  dependencies: [common_dep, mpi_dep],
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )

merge_shards = executable('merge_shards', 'merge_shards.cpp',
  dependencies: [common_dep, mpi_dep],
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )


sort_shard = executable('sort_shard', files('sort_shard.cpp'),
    dependencies: common_dep,
    include_directories: basis_inc,
    cpp_args: target_args,
    install: true
    )

sort_shard = executable('merge_sorted_shards', files('merge_sorted_shards.cpp'),
    dependencies: [common_dep, hdf5_dep],
    include_directories: basis_inc,
    cpp_args: target_args,
    install: true
    )

partition_shard = executable('partition_shard', files('partition_shard.cpp'),
    dependencies: common_dep,
    include_directories: basis_inc,
    cpp_args: target_args,
    install: true
    )
