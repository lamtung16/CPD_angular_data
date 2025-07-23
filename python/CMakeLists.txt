# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Include Pybind11 (handled by scikit-build)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Set Armadillo build options before adding the subdirectory
set(STATIC_LIB ON CACHE BOOL "Build Armadillo as a static library")
set(HEADER_ONLY OFF CACHE BOOL "Do not build Armadillo as header-only")

# Add CArma as a subdirectory 
add_subdirectory(ext/carma)


# Add the ruptures_apart_cpp library
get_filename_component(CPP_ABS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cpp" ABSOLUTE) # path to source files
add_subdirectory(${CPP_ABS_PATH} ${CMAKE_CURRENT_BINARY_DIR}/cpp) # out-of-tree source a binary directory must be explicitly specified.

# Build Python bindings
pybind11_add_module(_core src/main.cpp)
target_include_directories(_core PRIVATE
    ${PROJECT_SOURCE_DIR}/ext/carma
    ${CPP_ABS_PATH}/src
)
target_link_libraries(_core PRIVATE armadillo carma::carma ruptures_apart_cpp)

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION ruptures_apart)