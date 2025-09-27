# the important tests
spinon_tests = find_program('test_spinonfree_222.sh')
test('gen_spinon_basis_st', spinon_tests, args: ['single'])
test('gen_spinon_basis_par', spinon_tests, args: ['parallel'])
test('gen_spinon_basis_spinorder', spinon_tests, args: ['spinorder'])


test_uint128 = executable('test_uint128',
  'test_uint128.cpp',
  include_directories: inc,
  install : false)


test_PMR_operator = executable('test_PMR_operator',
  'test_PMR_construct.cpp',
  include_directories: inc,
  dependencies: [eigen_dep, hdf5_dep],
  install : false)

test_lanczos = executable('test_lanczos',
  'test_lanczos.cpp',
  include_directories: inc,
  dependencies: g_deps,
  install: false
  )

#test_ftlm_ops = executable('test_ftlm_ops','test_ftlm_ops.cpp',
#                         include_directories: inc,
#                         dependencies: [eigen_dep],
#                         install : false)


# Register specific test categories
test('PMRoperator_edge_case', test_PMR_operator,
  timeout : 10,
  suite: ['unit', 'constructors']
  )

test('uint128_constructors', test_uint128,
     args : ['--test-constructors'],
     timeout : 10,
     suite : ['unit', 'constructors'])

test('uint128_operators', test_uint128,
     args : ['--test-operators'],
     timeout : 10,
     suite : ['unit', 'operators'])

test('uint128_bit_operations', test_uint128,
     args : ['--test-bit-ops'],
     timeout : 10,
     suite : ['unit', 'bit_operations'])

test('uint128_hash_equality', test_uint128,
     args : ['--test-hash'],
     timeout : 10,
     suite : ['unit', 'hash'])

test('uint128_edge_cases', test_uint128,
     args : ['--test-edge-cases'],
     timeout : 10,
     suite : ['unit', 'edge_cases'])


test('test_lanczos', test_lanczos)
#test('Krylov operations', test_ftlm_ops)
