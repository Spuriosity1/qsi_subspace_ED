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



set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Find HDF5 (both C and C++ versions)
find_package(HDF5 REQUIRED COMPONENTS C CXX)


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


add_subdirectory("test")

find_package(nlohmann_json 3.11.0 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Spectra 1.1.0 REQUIRED)


# Basis building
target_link_libraries(gen_spinon_basis PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(gen_spinon_basis PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(gen_spinon_basis PRIVATE ${HDF5_C_LIBS} ${HDF5_CXX_LIBS})

target_include_directories(gen_spinon_basis_parallel PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(gen_spinon_basis_parallel PRIVATE ${HDF5_C_LIBS} ${HDF5_CXX_LIBS})
target_link_libraries(gen_spinon_basis_parallel PRIVATE nlohmann_json::nlohmann_json)

# Hamiltonian building and operator classes


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -g -O0 -D_GLIBCXX_USE_CXX11_ABI=1")
    target_compile_definitions(gen_spinon_basis_parallel PRIVATE VERBOSITY=3)
    target_compile_definitions(gen_spinon_basis PRIVATE VERBOSITY=3)
elseif(CMAKE_BUILD_TYPE STREQUAL "QuietDebug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -g -O1 -D_GLIBCXX_USE_CXX11_ABI=1")
    target_compile_definitions(gen_spinon_basis_parallel PRIVATE VERBOSITY=0)
    target_compile_definitions(gen_spinon_basis PRIVATE VERBOSITY=0)
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -Ofast")
endif()
    
