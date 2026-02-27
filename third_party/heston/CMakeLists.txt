cmake_minimum_required(VERSION 3.18)
project(heston LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# User-configurable options
option(USE_MKL "Use Intel MKL instead of CLAPACK" ON)
if(USE_MKL) 
    file(TO_CMAKE_PATH "$ENV{MKLROOT}" LAPACK_DIR_DEFAULT)
else()
    file(TO_CMAKE_PATH "$ENV{DEV}/cpp/clapack-3.2.1" LAPACK_DIR_DEFAULT)
endif()

file(TO_CMAKE_PATH "$ENV{DEV}/cpp/eigen-5.0.0" EIGEN_DIR_DEFAULT)
file(TO_CMAKE_PATH "$ENV{DEV}/cpp/levmar-2.6" LEVMAR_DIR_DEFAULT)

if(NOT DEFINED EIGEN_DIR)
    set(EIGEN_DIR "${EIGEN_DIR_DEFAULT}")
endif()
if(NOT DEFINED LEVMAR_DIR)
    set(LEVMAR_DIR "${LEVMAR_DIR_DEFAULT}")
endif()
if(NOT DEFINED LAPACK_DIR)
    set(LAPACK_DIR "${LAPACK_DIR_DEFAULT}")
endif()

message(STATUS "USE_MKL:    ${USE_MKL}")
message(STATUS "EIGEN_DIR:  ${EIGEN_DIR}")
message(STATUS "LEVMAR_DIR: ${LEVMAR_DIR}")
message(STATUS "LAPACK_DIR: ${LAPACK_DIR}")

# Check existence
if(NOT EXISTS "${EIGEN_DIR}")
    message(FATAL_ERROR "EIGEN_DIR not found: ${EIGEN_DIR}")
endif()

if(NOT EXISTS "${LEVMAR_DIR}")
    message(FATAL_ERROR "LEVMAR_DIR not found: ${LEVMAR_DIR}")
endif()

if(NOT EXISTS "${LAPACK_DIR}")
    message(FATAL_ERROR "LAPACK_DIR not found: ${LAPACK_DIR}")
endif()

add_library(heston SHARED
    src/Bindings.cpp
    src/HestonTermStructure.cpp
    src/HestonCalibrator.cpp
)


# Includes
target_include_directories(
    heston PRIVATE
    ${EIGEN_DIR}
    ${LEVMAR_DIR}/src
)

# Link levmar
find_library(LEVMAR_LIB levmar PATHS ${LEVMAR_DIR}/build/Release REQUIRED)
target_link_libraries(heston PRIVATE ${LEVMAR_LIB})

# Configure LAPACK backend
if(USE_MKL)
    target_include_directories(heston PRIVATE ${LAPACK_DIR}/include)
    find_library(MKL_INTEL_LIB mkl_intel_lp64 PATHS ${LAPACK_DIR}/lib REQUIRED)
    find_library(MKL_SEQUENTIAL_LIB mkl_sequential PATHS ${LAPACK_DIR}/lib REQUIRED)
    find_library(MKL_CORE_LIB mkl_core PATHS ${LAPACK_DIR}/lib REQUIRED)
    
    target_link_libraries(heston PRIVATE 
        ${MKL_INTEL_LIB}
        ${MKL_SEQUENTIAL_LIB}
        ${MKL_CORE_LIB}
    )
else()
    target_include_directories(heston PRIVATE ${LAPACK_DIR}/include)
    find_library(BLAS_LIB blas PATHS ${LAPACK_DIR}/lib REQUIRED)
    find_library(LAPACK_LIB lapack PATHS ${LAPACK_DIR}/lib REQUIRED)
    find_library(F2C_LIB f2c libf2c PATHS ${LAPACK_DIR}/lib REQUIRED)
    
    target_link_libraries(heston PRIVATE 
        ${LAPACK_LIB}
        ${BLAS_LIB}
        ${F2C_LIB}
    )
endif()

# Outputs
set_target_properties(heston PROPERTIES
    PREFIX ""
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
