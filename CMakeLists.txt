cmake_minimum_required(VERSION 3.19)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

project(
    subspace_ED
)

include_directories(
    include
    vendor/include
)


# Custom function to find packages. First try config mode. If the target isn't found,
# use module mode.
function(find_package_custom name target)
  find_package(${name} CONFIG QUIET)
  
  if (NOT TARGET ${target})
    find_package(${name} REQUIRED)
  else()
    get_target_property(location ${target} LOCATION)
    message( STATUS "Found ${name} [config]: ${location}" )
  endif()
endfunction()



# In case dependencies were installed by a package manager, we need to let CMake know
# where to look for the modules.
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules)


#find_package_custom(arpackng ARPACK::ARPACK)
#find_package_custom(superlu superlu::superlu)
#find_package(arpackng)

# the tests
# add_executable(benchmark_uints test/benchmark_uints.cpp)

#add_executable(test_arpack test/test_arpack.cpp)
#target_include_directories(test_arpack PUBLIC ARPACK::ARPACK)
#target_link_libraries(test_arpack ARPACK::ARPACK superlu::superlu)
#
#
#add_executable(test_arpack_SLU test/test_arpack_SLU.cpp)
#target_include_directories(test_arpack_SLU PUBLIC ARPACK::ARPACK)
#target_link_libraries(test_arpack_SLU ARPACK::ARPACK superlu::superlu)

# main functions
add_executable(gen_spinon_basis
    src/gen_spinon_basis.cpp
    src/tetra_graph_io.cpp
    src/pyro_tree.cpp
)

add_executable(gen_spinon_basis_parallel
    src/gen_spinon_basis_parallel.cpp
    src/tetra_graph_io.cpp
    src/pyro_tree.cpp
)

#add_executable(build_hamiltonian
#    src/build_hamiltonian.cpp
#    src/tetra_graph_io.cpp
#)
#
#target_include_directories(build_hamiltonian PUBLIC ARPACK::ARPACK)
#target_link_libraries(build_hamiltonian ARPACK::ARPACK superlu::superlu)
#




# TARGET_INCLUDE_DIRECTORIES(build_hamiltonian PUBLIC PARPACK::PARPACK)
# TARGET_LINK_LIBRARIES(build_hamiltonian::PARPACK)
# TARGET_INCLUDE_DIRECTORIES(build_hamiltonian PUBLIC MPI::MPI_Fortran)
# TARGET_LINK_LIBRARIES(build_hamiltonian MPI::MPI_Fortran)


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "-DNDEBUG -Ofast")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wpedantic -g -O0")
    target_compile_definitions(gen_spinon_basis_parallel PRIVATE VERBOSITY=2)
    target_compile_definitions(gen_spinon_basis PRIVATE VERBOSITY=2)

endif()
    
