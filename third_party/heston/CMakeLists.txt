cmake_minimum_required(VERSION 3.18)
project(heston LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

file(GLOB_RECURSE HESTON_SOURCES src/*.cpp src/*.cxx src/*.cc)
add_library(heston SHARED ${HESTON_SOURCES})
target_compile_definitions(heston PRIVATE BUILDING_HESTON_DLL)

# Includes
target_include_directories(
    heston PRIVATE
    "C:/dev/src/eigen-5.0.0"
    "C:/dev/src/levmar-2.6"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include"
)

# Link libraries
target_link_libraries(
    heston PRIVATE
    "C:/dev/build/levmar-2.6/Release/levmar.lib"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.2/lib/mkl_intel_lp64.lib"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.2/lib/mkl_sequential.lib"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.2/lib/mkl_core.lib"
)

# Outputs
set_target_properties(heston PROPERTIES
    PREFIX ""
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "Configured heston shared library.")