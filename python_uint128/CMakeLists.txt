cmake_minimum_required(VERSION 3.19)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

## Python uint128's

# Fetch pybind11 if not provided
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0  # or latest stable
)
FetchContent_MakeAvailable(pybind11)

# Get Python's site-packages directory for the virtual environment
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set the output directory for the Python extension module
set(PYTHON_MODULE_DIR "${PYTHON_SITE_PACKAGES}")

# Build the Python extension
pybind11_add_module(uint128 MODULE uint128.cpp)

# You can link extra flags if needed
target_compile_features(uint128 PRIVATE cxx_std_17)
target_compile_options(uint128 PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

# Set the output directory for the module to the virtual environment's site-packages
set_target_properties(uint128 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_MODULE_DIR}"
)
