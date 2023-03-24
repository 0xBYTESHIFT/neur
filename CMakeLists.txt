cmake_minimum_required(VERSION 3.18.0)
project(neur VERSION 0.1.4)

set( CMAKE_CXX_STANDARD 17 )

include(cmake/CPM.cmake)
include(CheckCXXCompilerFlag)

include_directories(include)

set(CPM_SOURCE_CACHE ".CPM")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(NEUR_USE_SIMD "use xsimd" ON)
option(NEUR_USE_JSON "use json loading/saving" ON)
option(NEUR_BUILD_TESTS "build tests" ON)

if(NOT DEFINED BLAS_VENDOR)
  set(BLAS_VENDOR OpenBLAS)
endif()

find_package(BLAS REQUIRED)

CPMAddPackage("gh:xtensor-stack/xtl#0.7.5")
CPMAddPackage("gh:xtensor-stack/xtensor#0.24.4")
CPMAddPackage("gh:xtensor-stack/xtensor-blas#0.20.0")
CPMAddPackage("gh:gabime/spdlog#v1.11.0")

if( NEUR_BUILD_TESTS )
    CPMAddPackage("gh:google/googletest#release-1.12.1")
endif()

if( NEUR_USE_SIMD )
    CPMAddPackage("gh:xtensor-stack/xsimd#8.0.3")
endif()
if( NEUR_USE_JSON )
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    CPMAddPackage("gh:nlohmann/json#v3.11.2")
    include_directories(${nlohmann_json_SOURCE_DIR}/single_include)
endif()

# library
add_library(neur INTERFACE)
target_include_directories(neur INTERFACE
    include/
    ${spdlog_SOURCE_DIR}/include
    ${xtensor-blas_SOURCE_DIR}/include
) 
if( NEUR_USE_SIMD )
    target_compile_definitions(neur INTERFACE NEUR_SIMD=1)
endif()

target_link_libraries(neur INTERFACE
    xtl xtensor xtensor-blas
    BLAS::BLAS
    spdlog
)
if( NEUR_USE_JSON )
    target_link_libraries(neur INTERFACE nlohmann_json)
endif()
if( NEUR_USE_SIMD )
    target_link_libraries(neur INTERFACE xsimd)
endif()

# === tests ===
if( NOT NEUR_BUILD_TESTS )
    return()
endif()

include(CTest)
enable_testing()

add_executable(brain_test tests/test_brain_common.cpp)
target_link_libraries(brain_test neur 
    gtest gtest_main gmock)

add_test(brain_test brain_test)

if( NEUR_USE_JSON )
    add_executable(brain_io_test tests/test_brain_io.cpp)
    target_link_libraries(brain_io_test neur 
        gtest gtest_main gmock)
    add_test(brain_io_test brain_io_test)
endif()
