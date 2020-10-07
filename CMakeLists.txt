cmake_minimum_required(VERSION 3.11.0)
project(brain VERSION 0.1.0)

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
FetchContent_Declare(xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd
)
FetchContent_Declare(xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl
)
FetchContent_Declare(xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
)
FetchContent_MakeAvailable(xtl)
FetchContent_MakeAvailable(xtensor)
FetchContent_MakeAvailable(xsimd)

add_library(neur INTERFACE)
target_include_directories(neur INTERFACE include/) 

add_executable(brain_test main.cpp)
target_link_libraries(brain_test
    OpenBLAS::OpenBLAS
    neur
)

# add_executable(test_chromosome tests/test_chromosome.cpp)
# add_test(test_chromosome test_chromosome)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)