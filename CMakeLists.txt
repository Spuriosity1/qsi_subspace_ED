cmake_minimum_required(VERSION 3.19)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

project(
    subspace_ED
)

# find_package(xdiag REQUIRED)
find_package(nlohmann_json 3.11.3 REQUIRED)


set(CMAKE\_BUILD\_TYPE Debug)
if(NOT ${CMAKE\_BUILD\_TYPE} STREQUAL "Debug" OR NOT ${CMAKE\_BUILD\_TYPE} STREQUAL "Release")
set(CMAKE\_BUILD\_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "-DNDEBUG -Ofast")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wpedantic -g -O0")
endif()
    
include_directories(include)

# the tests
# add_executable(benchmark_uints test/benchmark_uints.cpp)

# main function
add_executable(gen_spinon_basis
    src/gen_spinon_basis.cpp
    src/tetra_graph_io.cpp
    src/pyro_tree.cpp
)
target_link_libraries(gen_spinon_basis PRIVATE nlohmann_json::nlohmann_json)
