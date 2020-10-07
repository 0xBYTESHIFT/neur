cmake_minimum_required(VERSION 3.11.0)
project(neur VERSION 0.1.1)

include(CTest)
enable_testing()
include_directories(include)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -s -DNDEBUG -march=native")
set(BLAS_VENDOR OpenBLAS)
find_package(OpenBLAS QUIET)
set(BLAS_LIB OpenBLAS_LIB)
set(BLAS_DIR OpenBLAS_DIR)
set(BLAS_INCLUDE_DIR OpenBLAS_INCLUDE_DIR)
if(NOT OpenBLAS_FOUND)
    message(FATAL_ERROR "no openblas")
endif()

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
    endif(NOT ${${package}_FOUND})
endmacro()

find_fetch(xtl https://github.com/xtensor-stack/xtl)
find_fetch(xsimd https://github.com/xtensor-stack/xsimd)
find_fetch(xtensor https://github.com/xtensor-stack/xtensor)
find_fetch(xtensor-blas https://github.com/xtensor-stack/xtensor-blas)

add_library(neur INTERFACE)
target_include_directories(neur INTERFACE include/) 

add_executable(brain_test main.cpp)
target_link_libraries(brain_test
    OpenBLAS::OpenBLAS
    neur
    xtl
    xsimd
    xtensor
    xtensor-blas
)

# add_executable(test_chromosome tests/test_chromosome.cpp)
# add_test(test_chromosome test_chromosome)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)