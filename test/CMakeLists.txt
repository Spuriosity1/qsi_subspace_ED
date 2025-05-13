
add_executable(test_basis_io
	test_basis_io.cpp
    ../src/tetra_graph_io.cpp
    ../src/pyro_tree.cpp

)


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


find_package(nlohmann_json 3.11.0 REQUIRED)

target_link_libraries(     test_basis_io PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(test_basis_io PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(     test_basis_io PRIVATE ${HDF5_C_LIBS} ${HDF5_CXX_LIBS})
