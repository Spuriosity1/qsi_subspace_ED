# umap_memtest = executable('umap_memtest', 'src/umap_memtest.cpp', include_directories: inc)
# umap_memtest = executable('map_memtest', 'src/map_memtest.cpp', include_directories: inc)
# umap_memtest = executable('vector_memtest', 'src/vector_memtest.cpp', include_directories: inc)
# 
# 
# udense_dep = dependency('unordered_dense', required:false)
# 
# if udense_dep.found()
#   umap_memtest = executable('ankerl_memtest', 'src/ankerl_memtest.cpp', include_directories: inc, dependencies: udense_dep)
#   map_choice = executable('map_choice', 'src/map_choice.cpp', include_directories: inc, dependencies: udense_dep)
# endif




# Benchmark test (if you want performance testing)
benchmark_uint128 = executable('benchmark_uint128',
                              'benchmark_uint128.cpp',
                              include_directories: inc,
                              install : false)

# Register benchmark (optional)
benchmark('uint128_performance', benchmark_uint128,
         timeout : 60,
         suite : 'benchmark')
