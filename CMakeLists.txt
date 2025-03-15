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


# Find HDF5 (both C and C++ versions)
find_package(HDF5 REQUIRED COMPONENTS C CXX)


# Debugging output
message(STATUS "HDF5 Include Dirs: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 Libraries: ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES}")

# Ensure we use the correct libraries in case imported targets are missing
set(HDF5_C_LIBS "")
set(HDF5_CXX_LIBS "")

if(HDF5_C_LIBRARIES)
    set(HDF5_C_LIBS ${HDF5_C_LIBRARIES})
elseif(TARGET HDF5::HDF5_C)
    set(HDF5_C_LIBS HDF5::HDF5_C)
endif()

if(HDF5_CXX_LIBRARIES)
    set(HDF5_CXX_LIBS ${HDF5_CXX_LIBRARIES})
elseif(TARGET HDF5::HDF5_CXX)
    set(HDF5_CXX_LIBS HDF5::HDF5_CXX)
endif()

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

add_executable(csv_to_h5
    src/csv_to_h5.cpp
)

#add_executable(build_hamiltonian
#    src/build_hamiltonian.cpp
#    src/tetra_graph_io.cpp
#)
#
#target_include_directories(build_hamiltonian PUBLIC ARPACK::ARPACK)
#target_link_libraries(build_hamiltonian ARPACK::ARPACK superlu::superlu)
#


target_include_directories(gen_spinon_basis PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(gen_spinon_basis PRIVATE ${HDF5_C_LIBS} ${HDF5_CXX_LIBS})

target_include_directories(gen_spinon_basis_parallel PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(gen_spinon_basis_parallel PRIVATE ${HDF5_C_LIBS} ${HDF5_CXX_LIBS})



# TARGET_INCLUDE_DIRECTORIES(build_hamiltonian PUBLIC PARPACK::PARPACK)
# TARGET_LINK_LIBRARIES(build_hamiltonian::PARPACK)
# TARGET_INCLUDE_DIRECTORIES(build_hamiltonian PUBLIC MPI::MPI_Fortran)
# TARGET_LINK_LIBRARIES(build_hamiltonian MPI::MPI_Fortran)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "-Wall -Wpedantic -g -O0 -D_GLIBCXX_USE_CXX11_ABI=1")
    target_compile_definitions(gen_spinon_basis_parallel PRIVATE VERBOSITY=2)
    target_compile_definitions(gen_spinon_basis PRIVATE VERBOSITY=2)
else()
    set(CMAKE_CXX_FLAGS "-DNDEBUG -Ofast")
endif()
    
