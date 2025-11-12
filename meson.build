project('projected_ed', ['cpp', 'c'],
  version: '1.0.0',
  meson_version : '>= 1.7.0',
  default_options : [ 'warning_level=1', 'buildtype=release', 'cpp_std=c++20' ]
)

cmake = import('cmake')

# add_project_arguments(['
add_project_link_arguments(['-pthread'], language: 'cpp' )

# Compiler flags for 128-bit integer support
cpp_compiler = meson.get_compiler('cpp')

host_cpu = host_machine.cpu_family()

target_args = []

if get_option('buildtype') == 'release'
  target_args += ['-DNDEBUG'] 
  if host_cpu.startswith('x86')
    if cpp_compiler.has_argument('-mavx2')
      target_args += ['-mavx2']
    elif cpp_compiler.has_argument('-mavx')
      target_args += ['-mavx']
    endif
    if cpp_compiler.has_argument('-msse2')
      target_args += ['-msse2']
    endif
  elif host_cpu.startswith('aarch64') or host_cpu.startswith('arm')
    # Apple silicon, we need to juggle a bit 
    #    if cpp_compiler.has_argument('-mfpu=neon')
    #      target_args += ['-mfpu=neon']
    #    elif cpp_compiler.has_argument('-march=armv8-a+simd')
    #      target_args += ['-march=armv8-a+simd']
    #    endif
    #    if cpp_compiler.has_argument('-march=armv8-a+sha3')
    #      target_args += ['-march=armv8-a+sha3']
    #    endif
  endif

  target_args += ['-DNDEBUG']

elif get_option('buildtype') == 'debug'
#   add_project_arguments(['-fsanitize=address'], language: 'cpp' )
endif


blas_backend = get_option('blas_backend')  # 'auto', 'openblas', 'mkl', 'generic'

# Apple bullshit
# For christ sake just use a goddamn normal header file you turtleneck wearing
# hipster shits I swear to god
accelerate_dep = declare_dependency(
  compile_args: ['-framework', 'Accelerate',
    '-DUSE_APPLE_ACCELERATE', 
    '-DACCELERATE_NEW_LAPACK',
    '-DACCELERATE_LAPACK_ILP64'],
    link_args: ['-framework', 'Accelerate'],
  )

if host_cpu.startswith('aarch64') and blas_backend != 'Accelerate'
  message('Trying to link with a non-Accelerate BLAS backend on macOS. This will probably not work!')
endif

if blas_backend == 'Accelerate'
  blas_dep = accelerate_dep
  lapack_dep = accelerate_dep

message('compiler is ', cpp_compiler.get_id())
  if cpp_compiler.get_id() == 'gcc'
    add_project_arguments('-flax-vector-conversions', language: 'cpp')
  endif
elif blas_backend == 'mkl'
  mkl_dep = dependency('mkl-dynamic-lp64-seq', required: false)
  if not mkl_dep.found()
    # Fallback: manual MKL detection
    mkl_root = get_option('mkl_root')
    if mkl_root == ''
      mkl_root = run_command('printenv', 'MKLROOT', check: false).stdout().strip()
    endif
    
    if mkl_root != ''
      mkl_inc = include_directories(mkl_root / 'include')
      mkl_lib = meson.get_compiler('cpp').find_library('mkl_intel_lp64',
        dirs: [mkl_root / 'lib' / 'intel64'])
      mkl_thread = meson.get_compiler('cpp').find_library('mkl_gnu_thread',
        dirs: [mkl_root / 'lib' / 'intel64'])
      mkl_core = meson.get_compiler('cpp').find_library('mkl_core',
        dirs: [mkl_root / 'lib' / 'intel64'])
      
      mkl_dep = declare_dependency(
        include_directories: mkl_inc,
        dependencies: [mkl_lib, mkl_thread, mkl_core],
        compile_args: ['-DMKL_ILP64']
      )
    else
      error('MKL requested but MKLROOT not set')
    endif
  endif
  blas_dep = mkl_dep
  lapack_dep = mkl_dep
  
elif blas_backend == 'openblas'
  # OpenBLAS (usually has cblas built-in)
  blas_dep = dependency('openblas', required: true)
  lapack_dep = blas_dep
  
elif blas_backend == 'generic' or blas_backend == 'auto'
  
  # Try to find any BLAS implementation
  blas_dep = dependency('cblas', required: false)
  if not blas_dep.found()
    blas_dep = dependency('blas', required: false)
  endif
  if not blas_dep.found()
    # Manual search
    cblas_lib = meson.get_compiler('cpp').find_library('cblas', required: false)
    blas_lib = meson.get_compiler('cpp').find_library('blas', required: false)
    if cblas_lib.found() and blas_lib.found()
      blas_dep = declare_dependency(dependencies: [cblas_lib, blas_lib])
    else
      error('BLAS not found. Install BLAS/CBLAS or use -Dblas_backend=mkl')
    endif
  endif

  lapack_dep = dependency('lapack', required: false)
  if not lapack_dep.found()
    lapack_lib = cc.find_library('lapack', required: false)
    if lapack_lib.found()
      lapack_dep = lapack_lib
    else
      # Some systems (like Debian/Ubuntu) fold LAPACK into libblas
      lapack_dep = blas_dep
    endif
  endif
  
endif

message('Using BLAS backend: @0@'.format(blas_backend))



mpi_dep = dependency('mpi', language: 'cpp', required:true)

hdf5_raw_dep = dependency('hdf5',
  language: 'c',
  required: false,
  not_found_message: 'dependency HDF5 not found')


if hdf5_raw_dep.found()

  hdf5_is_parallel = false
  # Check if h5pcc exists and supports parallel
  
  # Check for parallel support
  cpp = meson.get_compiler('cpp')
  parallel_test = '''
  #include <H5pubconf.h>
  #ifndef H5_HAVE_PARALLEL
  #error "No parallel"
  #endif
  int main() { return 0; }
  '''

  hdf5_is_parallel = cpp.compiles(parallel_test, 
    dependencies: hdf5_raw_dep,
    name: 'HDF5 parallel support')

  h5_dep_dep = [hdf5_raw_dep]
  if hdf5_is_parallel
    message('HDF5 has parallel support')
    h5_dep_dep += mpi_dep
  else
    error('HDF5 does not have parallel support')
  endif

  hdf5_dep = declare_dependency(
    dependencies: h5_dep_dep
    )
endif




json_dep = dependency('nlohmann_json', required: true)

eigen_dep = dependency('Eigen3', method: 'cmake',  modules: ['Eigen3::Eigen'], required: true).as_system()

spectra_dep = dependency('spectra', method: 'cmake')

omp_dep = dependency('OpenMP', required: true)


cmake_opts = cmake.subproject_options()
cmake_opts.add_cmake_defines({
  'CMAKE_SOURCE_DIR': '/dummy/path/not/equal',
  'ENABLE_WERROR': 'OFF'
})


# Local include dir for headers like bittools.hpp, admin.hpp, basis_io

vendor_inc = include_directories('vendor/include')

inc = [ include_directories('include'), vendor_inc ]

g_deps = [
  hdf5_dep,
  json_dep,
  eigen_dep
]

subdir('components')
subdir('src')


polyring_basis = executable('polyring_basis', files('src/polyring_basis.cpp'), 
  dependencies: [common_dep, eigen_dep, operator_dep],
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )

polyring_basis_mpi = executable('polyring_basis_mpi', files('src/polyring_basis_mpi.cpp'), 
  dependencies: [common_dep, eigen_dep, operator_dep, mpi_dep],
  include_directories:basis_inc,
  cpp_args: target_args,
  install: true
  )

expectation_eval = static_library('expectation_eval', 'src/expectation_eval.cpp',
  include_directories: inc,
  dependencies: [operator_dep, linalg_routines_dep, g_deps])


diag_DOQSI_ham = executable('diag_DOQSI_ham',
  files(['src/diag_DOQSI_ham.cpp']),
  dependencies: [g_deps, eigen_dep, operator_dep, linalg_routines_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )

diag_DOQSI_ham_mpi = executable('diag_DOQSI_ham_mpi',
  files(['src/diag_DOQSI_ham_mpi.cpp']),
  dependencies: [g_deps, eigen_dep, operator_mpi_dep, linalg_routines_mpi_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )


#build_hamiltonian = executable('build_hamiltonian',
#  files(['src/build_hamiltonian.cpp', 'src/expectation_eval.cpp']),
#  dependencies: [g_deps, eigen_dep, spectra_dep],
#  include_directories: inc,
#  link_with: expectation_eval,
#  cpp_args: target_args
#  )

#ftlm_DOQSI_ham = executable('ftlm_DOQSI_ham',
#  files(['src/ftlm_DOQSI_ham.cpp', 'src/expectation_eval.cpp','src/ftlm.cpp']),
#  dependencies: [g_deps, eigen_dep, spectra_dep],
#  include_directories: inc,
#  cpp_args: target_args
#  )


eval_observables = executable('eval_observables',
  files(['src/eval_observables.cpp', 'src/expectation_eval.cpp']),
  dependencies: [g_deps, eigen_dep, omp_dep, operator_dep, linalg_routines_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )

ring_normed = executable('ring_normed',
  files(['src/ring_normed.cpp', 'src/expectation_eval.cpp']),
  dependencies: [g_deps, eigen_dep, omp_dep, operator_dep, linalg_routines_dep, spectra_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )


eval_observables_mpi = executable('eval_observables_mpi',
  files(['src/eval_observables_mpi.cpp', 'src/expectation_eval.cpp']),
  dependencies: [g_deps, eigen_dep, omp_dep, operator_mpi_dep, linalg_routines_dep],
  include_directories: inc,
  link_with: expectation_eval,
  cpp_args: target_args
  )
#
#
#eval_dsf = executable('eval_dsf',
#  files(['src/eval_dsf.cpp', 'src/expectation_eval.cpp']),
#  dependencies: [g_deps, eigen_dep, spectra_dep],
#  include_directories: inc,
#  link_with: expectation_eval,
#  cpp_args: target_args
#  )


if get_option('build_tests')
  subdir('test')
endif

if get_option('build_benchmarks')
  subdir('bench')
endif
