
cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(CxxWrapGCTest VERSION 0.01 LANGUAGES C CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_compile_options(
  -O3
  -march=native
  -funroll-loops
  -fopenmp
  -fPIC
  -Wall
  -Wextra
  -falign-functions=32
)

execute_process(
  COMMAND julia -e "println(DEPOT_PATH[1])"
  OUTPUT_VARIABLE JULIA_DEP_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(JlCxx_DIR "/home/david/.julia/dev/libcxxwrap_julia_jll/override")

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")


add_library(cxxwrapgctest SHARED main.cpp)

# both can be private I think
target_link_libraries(cxxwrapgctest PUBLIC JlCxx::cxxwrap_julia)

install(TARGETS cxxwrapgctest DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/cmake-install")
