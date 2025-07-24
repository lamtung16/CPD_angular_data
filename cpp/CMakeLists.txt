cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 20)

# Find and link BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
if(NOT BLAS_FOUND OR NOT LAPACK_FOUND)
    message(FATAL_ERROR "BLAS and LAPACK libraries are required but not found.")
endif()

# Collect all source files
file(GLOB CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# On Linux, static libraries without -fPIC cannot be linked into shared objects.
# This tells CMake to compile all source files with -fPIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add the source files to the library
add_library(ruptures_apart_cpp STATIC ${CPP_SOURCES})

# Include Armadillo
target_include_directories(ruptures_apart_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ext/armadillo/include)

# Link dependencies
target_link_libraries(ruptures_apart_cpp PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} armadillo)