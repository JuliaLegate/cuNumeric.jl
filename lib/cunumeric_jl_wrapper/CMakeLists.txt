cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(cuNumericWrapper)

set(cuNumericWrapperVersion 0.0.1)
message(STATUS "Project version: v${cuNumericWrapperVersion}")
set(CXX_CUNUMERICJL_WRAPPER cunumeric_jl_wrapper)
set(C_INTERFACE_LIB cunumeric_c_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(LEGATE_WRAPPER_ENABLE_CUDA "Build cunumeric_jl_wrapper with CUDA support" ON)
option(BINARYBUILDER "Building with binary builder" ON)

find_package(legate REQUIRED)
find_package(cupynumeric REQUIRED)

# CxxWrap Stuff
if(NOT BINARYBUILDER)
execute_process(
  COMMAND julia -e "println(DEPOT_PATH[1])"
  OUTPUT_VARIABLE JULIA_DEP_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(JlCxx_DIR "${JULIA_DEP_PATH}/dev/libcxxwrap_julia_jll/override")
message(STATUS "Setting JlCxx_DIR to ${JlCxx_DIR} (not using BinaryBuilder)")
endif()

find_package(JlCxx REQUIRED)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")

# Sources
set(SOURCES
    src/wrapper.cpp
    src/types.cpp
)

if(LEGATE_WRAPPER_ENABLE_CUDA)
    find_package(CUDAToolkit 13.0 REQUIRED)
    list(APPEND SOURCES src/cuda.cpp)
    message(STATUS "LEGATE_WRAPPER_ENABLE_CUDA=ON: adding src/cuda.cpp")
else()
    # only disables find_package requirement for CUDAToolkit.
    # if you have a CUDA enabled cuNumeric install, this really won't do anything.
    message(STATUS "LEGATE_WRAPPER_ENABLE_CUDA=OFF: skipping CUDAToolkit and src/cuda.cpp.")
endif()

# Library: C++ wrapper
add_library(${CXX_CUNUMERICJL_WRAPPER} SHARED ${SOURCES})
set_target_properties(${CXX_CUNUMERICJL_WRAPPER} PROPERTIES VERSION ${cuNumericWrapperVersion})

target_link_libraries(${CXX_CUNUMERICJL_WRAPPER} PRIVATE
    cupynumeric::cupynumeric
    legate::legate
    JlCxx::cxxwrap_julia
    JlCxx::cxxwrap_julia_stl
)

target_include_directories(${CXX_CUNUMERICJL_WRAPPER} PRIVATE include)
if(LEGATE_WRAPPER_ENABLE_CUDA)
    target_include_directories(${CXX_CUNUMERICJL_WRAPPER} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

install(TARGETS ${CXX_CUNUMERICJL_WRAPPER} DESTINATION lib)

# ---- C API ----
set(C_SOURCES
    src/ndarray.cpp
    src/memory.cpp
)

add_library(${C_INTERFACE_LIB} SHARED ${C_SOURCES})
set_target_properties(${C_INTERFACE_LIB} PROPERTIES VERSION ${cuNumericWrapperVersion})

target_link_libraries(${C_INTERFACE_LIB} PRIVATE
    cupynumeric::cupynumeric
    legate::legate
)

target_include_directories(${C_INTERFACE_LIB} PRIVATE include)
if(LEGATE_WRAPPER_ENABLE_CUDA)
    target_include_directories(${C_INTERFACE_LIB} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

install(TARGETS ${C_INTERFACE_LIB} DESTINATION lib)
