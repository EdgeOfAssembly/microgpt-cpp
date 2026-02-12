cmake_minimum_required(VERSION 3.20)
project(microgpt-cpp VERSION 1.0.0 LANGUAGES CXX)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Copy data files to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# Training example
add_executable(train examples/train.cpp)
target_include_directories(train PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Inference example
add_executable(infer examples/infer.cpp)
target_include_directories(infer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Install targets
install(DIRECTORY include/microgpt DESTINATION include)
install(TARGETS train infer DESTINATION bin)
