cmake_minimum_required(VERSION 3.13.0)
project(neur VERSION 0.1.2)

include(CTest)
enable_testing()
include_directories(include)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0079 NEW)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -s -DNDEBUG -march=native")

option(USE_BLAS "use openblas" OFF)
option(USE_SIMD "use xsimd" ON)

include(FetchContent)
include(CheckCXXCompilerFlag)

macro(find_fetch package repo)
    find_package(${package} QUIET)
    if(NOT ${${package}_FOUND})
        message(STATUS "fetching ${package} from ${repo}")
        FetchContent_Declare(${package}
            GIT_REPOSITORY ${repo}
        )
        FetchContent_MakeAvailable(${package})
        FetchContent_GetProperties(${package})
        if(NOT ${${package}_POPULATED})
          FetchContent_Populate(${package})
          add_subdirectory(${${package}_SOURCE_DIR} ${${package}_BINARY_DIR})
        endif(NOT ${${package}_POPULATED})
        list(APPEND CMAKE_PREFIX_PATH ${${package}_BINARY_DIR})
        find_package(${package} REQUIRED)
    endif(NOT ${${package}_FOUND})
endmacro()

find_fetch(xtl https://github.com/xtensor-stack/xtl)

if(USE_SIMD)
    set(XTENSOR_USE_XSIMD 1)
    find_fetch(xsimd https://github.com/xtensor-stack/xsimd)
endif()

find_fetch(xtensor https://github.com/xtensor-stack/xtensor)

if(USE_BLAS)
    set(BLAS_VENDOR OpenBLAS)
    find_package(OpenBLAS QUIET)
    set(BLAS_LIB OpenBLAS_LIB)
    set(BLAS_DIR OpenBLAS_DIR)
    set(BLAS_INCLUDE_DIR OpenBLAS_INCLUDE_DIR)
    if(NOT OpenBLAS_FOUND)
        message(FATAL_ERROR "no openblas")
    endif()
    find_fetch(xtensor-blas https://github.com/xtensor-stack/xtensor-blas)
else()
endif()

add_library(neur INTERFACE)
target_include_directories(neur INTERFACE include/) 

add_executable(brain_test main.cpp)
target_link_libraries(brain_test
    neur
    xtl
    xtensor
)

if(USE_SIMD)
    target_link_libraries(brain_test xsimd)
endif()

if(USE_BLAS)
    target_link_libraries(brain_test OpenBLAS::OpenBLAS xtensor-blas)
endif()

add_test(brain_test brain_test)

# add_executable(test_chromosome tests/test_chromosome.cpp)
# add_test(test_chromosome test_chromosome)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

unset(USE_BLAS)
unset(USE_SIMD)
unset(XTENSOR_USE_XSIMD)