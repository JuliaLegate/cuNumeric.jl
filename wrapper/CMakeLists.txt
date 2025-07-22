cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(cuNumericWrapper VERSION 0.01 LANGUAGES C CXX)

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
add_library(cupynumericwrapper SHARED
             src/wrapper.cpp
             src/types.cpp
             src/cuda.cpp)


set_target_properties(cupynumericwrapper PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${HDF5_PATH}/lib;${LEGATE_PATH}/lib;${CUPYNUMERIC_PATH}/lib"
)


set_target_properties(cupynumericwrapper PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${HDF5_PATH}/lib;${LEGATE_PATH}/lib;${CUPYNUMERIC_PATH}/lib"
)

# both can be private I think
target_link_libraries(cupynumericwrapper 
                        PRIVATE cupynumeric::cupynumeric
                        PRIVATE legate::legate
                        PUBLIC JlCxx::cxxwrap_julia
                      )


target_include_directories(cupynumericwrapper PRIVATE 
                            include
                            ${LEGATE_PATH}/include/legate
                            ${LEGATE_PATH}/include/legate/deps
                            ${CUDAToolkit_INCLUDE_DIRS}
                        )

install(TARGETS cupynumericwrapper 
          DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/cmake-install")


# C- API
add_library(cwrapper SHARED
    src/ndarray.cpp
)


set_target_properties(cwrapper PROPERTIES
BUILD_WITH_INSTALL_RPATH TRUE
INSTALL_RPATH "${HDF5_PATH}/lib;${LEGATE_PATH}/lib;${CUPYNUMERIC_PATH}/lib"
)

# both can be private I think
target_link_libraries(cwrapper 
           PRIVATE cupynumeric::cupynumeric
           PRIVATE legate::legate
         )


target_include_directories(cwrapper PRIVATE 
               include
               ${CUPYNUMERIC_PATH}/include
               ${LEGATE_PATH}/include/legate
               ${LEGATE_PATH}/include/legate/deps
               ${CUDAToolkit_INCLUDE_DIRS}
           )

install(TARGETS cwrapper 
DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/cmake-install")
