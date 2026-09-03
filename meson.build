project('projected_ed', ['cpp', 'c'],
  version: '1.0.0',
  meson_version : '>= 1.7.0',
  default_options : [ 'warning_level=1', 'buildtype=release', 'cpp_std=c++17' ]
)

cmake = import('cmake')

add_project_link_arguments(['-pthread','-lstdc++fs', '-lgomp'], language: 'cpp' )

# Compiler flags for 128-bit integer support
cpp_compiler = meson.get_compiler('cpp')

host_cpu = host_machine.cpu_family()

# target_args: per-target extras carried via `cpp_args: target_args`. Holds only
#   -DNDEBUG, so it stays off for unit tests that don't opt in (keeping their
#   assert()s live).
# simd_args: ISA + safe optimisation flags. These MUST be identical across every
#   C++ translation unit: Eigen derives EIGEN_MAX_ALIGN_BYTES per-TU from the
#   enabled vector ISA, so a TU that misses -march=znver5 (no __AVX512F__) links
#   against libs built with it and their aligned AVX-512 loads fault (SIGSEGV).
#   Applied project-wide via add_project_arguments below.
target_args = []
simd_args = []

cpu_arch = get_option('cpu_arch')

if get_option('buildtype') == 'release'
  target_args += ['-DNDEBUG']

  # Scalar/loop optimisation flags safe for the ED/Lanczos numerics: no
  # reassociation or fast-math (those can wreck Lanczos orthogonality).
  # Guarded so toolchains lacking them still configure.
  foreach flag : ['-funroll-loops', '-fno-math-errno', '-fno-trapping-math']
    if cpp_compiler.has_argument(flag)
      simd_args += [flag]
    endif
  endforeach

  if host_cpu.startswith('x86')
    if cpu_arch == 'auto'
      # Portable default: generic AVX2, runs on any recent x86 (dev boxes/CI).
      if cpp_compiler.has_argument('-mavx2')
        simd_args += ['-mavx2']
      elif cpp_compiler.has_argument('-mavx')
        simd_args += ['-mavx']
      endif
      if cpp_compiler.has_argument('-msse2')
        simd_args += ['-msse2']
      endif
    elif cpu_arch == 'generic'
      # No explicit ISA beyond the compiler baseline.
    else
      # Explicit target: 'native' or a named Zen microarchitecture.
      march = '-march=' + cpu_arch
      if cpp_compiler.has_argument(march)
        simd_args += [march, '-mtune=' + cpu_arch]
        # Zen 4 (double-pumped) / Zen 5 (native) have AVX-512; prefer wide
        # vectors. Benchmark this: on some parts 256-bit downclocks less.
        if cpu_arch == 'znver4' or cpu_arch == 'znver5'
          if cpp_compiler.has_argument('-mprefer-vector-width=512')
            simd_args += ['-mprefer-vector-width=512']
          endif
        endif
      else
        error('Compiler rejects -march=' + cpu_arch +
          ' (need AOCC 5+/clang 19+ for znver5, clang 16+ for znver4). ' +
          'Use -Dcpu_arch=auto or a supported target.')
      endif
    endif
  elif host_cpu.startswith('aarch64') or host_cpu.startswith('arm')
    # Apple silicon, we need to juggle a bit
    #    if cpp_compiler.has_argument('-mfpu=neon')
    #      simd_args += ['-mfpu=neon']
    #    elif cpp_compiler.has_argument('-march=armv8-a+simd')
    #      simd_args += ['-march=armv8-a+simd']
    #    endif
    #    if cpp_compiler.has_argument('-march=armv8-a+sha3')
    #      simd_args += ['-march=armv8-a+sha3']
    #    endif
  endif


elif get_option('buildtype') == 'debug'
#   add_project_arguments(['-fsanitize=address'], language: 'cpp' )
endif

# ISA/optimisation flags must be uniform across all C++ TUs (see the Eigen
# alignment note above); apply them project-wide. -DNDEBUG is deliberately NOT
# here — it stays in target_args so unit tests keep their assert()s enabled.
if simd_args.length() > 0
  add_project_arguments(simd_args, language: 'cpp')
endif


blas_backend = get_option('blas_backend')  # 'auto', 'openblas', 'mkl', 'aocl', 'generic'

# Apple bullshit
accelerate_dep = declare_dependency(
  compile_args: ['-framework', 'Accelerate',
    '-DUSE_APPLE_ACCELERATE', 
    '-DACCELERATE_NEW_LAPACK',
    '-DACCELERATE_LAPACK_ILP64'],
    link_args: ['-framework', 'Accelerate'],
  )

# AMD bullshit
cc = meson.get_compiler('c')

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
elif blas_backend == 'aocl'
  aocl_root = get_option('aocl_root')  

  blis_lib  = cc.find_library('blis-mt' )
  flame_lib = cc.find_library('flame'   )

  blas_dep = declare_dependency(
    compile_args: ['-DUSE_BLIS'],
    dependencies: blis_lib
  )
  lapack_dep = declare_dependency(
    dependencies: flame_lib
  )
elif blas_backend == 'mkl'
  mkl_dep = dependency('mkl-dynamic-lp64-seq', required: false)
  if mkl_dep.found()
    message('detected MKL')
  else
    message('manually detecting mkl')
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
  blas_dep = declare_dependency(compile_args: ['-DUSE_MKL'], dependencies: [mkl_dep])
  lapack_dep = blas_dep

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


# MPI and HDF5 
hdf5_root = get_option('hdf5_root')
mpi_root = get_option('mpi_root')

if mpi_root != ''
  message('Using custom MPI root: ' + mpi_root)

  mpi_inc = include_directories(join_paths(mpi_root, 'include'))
  mpi_libdir = join_paths(mpi_root, 'lib/release')
  mpi_dep = declare_dependency(
    include_directories: mpi_inc,
    link_args: [
      '-L' +mpi_libdir, '-lmpi']
    )
else
  mpi_dep = dependency('mpi', required: true, language: 'cpp')
endif

omp_dep = dependency('openmp', required: true, language: 'cpp')

if hdf5_root != ''
  message('Using custom HDF5 at ' + hdf5_root)

  hdf5_inc = include_directories(join_paths(hdf5_root, 'include'))
  hdf5_libdir = join_paths(hdf5_root, 'lib')

  hdf5_dep = declare_dependency(
    include_directories: hdf5_inc,
    link_args: ['-L' + hdf5_libdir, '-lhdf5'],
    dependencies: mpi_dep
  )
else
  hdf5_dep = declare_dependency(
		  dependencies: 
		  [dependency('hdf5', language: 'c', required: true),
		  mpi_dep])
endif



json_dep = dependency('nlohmann_json', required: true).as_system()

eigen_dep = dependency('Eigen3', method: 'cmake',  modules: ['Eigen3::Eigen'], required: true).as_system()

spectra_dep = dependency('Spectra',
  method: 'cmake',
  modules: ['Spectra::Spectra'],
  required: false
)
if not spectra_dep.found()
  spectra_dep = dependency('spectra', fallback: ['spectra', 'spectra_dep'])
endif


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

if get_option('build_tests')
  subdir('test')
endif

if get_option('build_benchmarks')
  subdir('bench')
endif
