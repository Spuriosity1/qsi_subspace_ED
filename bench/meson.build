
umap_memtest = executable('umap_memtest', 'src/umap_memtest.cpp', include_directories: inc)
umap_memtest = executable('map_memtest', 'src/map_memtest.cpp', include_directories: inc)
umap_memtest = executable('vector_memtest', 'src/vector_memtest.cpp', include_directories: inc)

udense_dep = dependency('unordered_dense', required:true)

umap_memtest = executable('ankerl_memtest', 'src/ankerl_memtest.cpp', include_directories: inc, dependencies: udense_dep)
