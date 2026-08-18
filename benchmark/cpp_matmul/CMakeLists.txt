
cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(cuNumericWrapper VERSION 0.01 LANGUAGES C CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

find_package(cupynumeric REQUIRED)

add_executable(matmulfp32_test main.cpp)
target_link_libraries(matmulfp32_test PRIVATE cupynumeric::cupynumeric)
install(TARGETS matmulfp32_test DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/cmake-install")
