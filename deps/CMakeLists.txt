cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(cuNumericWrapper VERSION 0.01 LANGUAGES C CXX)

set(CXX_CUNUMERICJL_WRAPPER cunumeric_jl_wrapper)
set(C_INTERFACE_LIB cunumeric_c_wrapper)

# Specify C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Ensure the LEGATE_PATH is defined
if(NOT DEFINED LEGATE_PATH)
    message(FATAL_ERROR "LEGATE_PATH is not set.")
endif()

if(NOT DEFINED HDF5_PATH)
    message(FATAL_ERROR "HDF5_PATH is not set.")
endif()

if(NOT DEFINED CUPYNUMERIC_PATH)
    message(FATAL_ERROR "CUPYNUMERIC_PATH is not set.")
endif()

find_package(CUDAToolkit 12.2 REQUIRED)
find_package(cupynumeric REQUIRED)

execute_process(
  COMMAND julia -e "println(DEPOT_PATH[1])"
  OUTPUT_VARIABLE JULIA_DEP_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# THERE IS NO INCLUDE DIR HERE???
set(JlCxx_DIR "${JULIA_DEP_PATH}/dev/libcxxwrap_julia_jll/override")
find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")


# Julia Wrapper Library
add_library(${CXX_CUNUMERICJL_WRAPPER} SHARED
             src/wrapper.cpp
             src/types.cpp
             src/cuda.cpp)


set_target_properties(${CXX_CUNUMERICJL_WRAPPER} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${HDF5_PATH}/lib;${LEGATE_PATH}/lib;${CUPYNUMERIC_PATH}/lib"
)


set_target_properties(${CXX_CUNUMERICJL_WRAPPER} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${HDF5_PATH}/lib;${LEGATE_PATH}/lib;${CUPYNUMERIC_PATH}/lib"
)

# both can be private I think
target_link_libraries(${CXX_CUNUMERICJL_WRAPPER} 
                        PRIVATE cupynumeric::cupynumeric
                        PRIVATE legate::legate
                        PUBLIC JlCxx::cxxwrap_julia
                      )


target_include_directories(${CXX_CUNUMERICJL_WRAPPER} PRIVATE 
                            include
                            ${LEGATE_PATH}/include/legate
                            ${LEGATE_PATH}/include/legate/deps
                            ${CUDAToolkit_INCLUDE_DIRS}
                        )
          
install(TARGETS ${CXX_CUNUMERICJL_WRAPPER} 
          DESTINATION "${PROJECT_INSTALL_PATH}/lib")

# C- API
add_library(${C_INTERFACE_LIB} SHARED
    src/ndarray.cpp
)


set_target_properties(${C_INTERFACE_LIB} PROPERTIES
BUILD_WITH_INSTALL_RPATH TRUE
INSTALL_RPATH "${HDF5_PATH}/lib;${LEGATE_PATH}/lib;${CUPYNUMERIC_PATH}/lib"
)

# both can be private I think
target_link_libraries(${C_INTERFACE_LIB} 
           PRIVATE cupynumeric::cupynumeric
           PRIVATE legate::legate
         )


target_include_directories(${C_INTERFACE_LIB} PRIVATE 
               include
               ${CUPYNUMERIC_PATH}/include
               ${LEGATE_PATH}/include/legate
               ${LEGATE_PATH}/include/legate/deps
               ${CUDAToolkit_INCLUDE_DIRS}
           )

install(TARGETS ${C_INTERFACE_LIB} 
          DESTINATION "${PROJECT_INSTALL_PATH}/lib")