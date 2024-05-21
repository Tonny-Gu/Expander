cmake_minimum_required(VERSION 3.10)
project(btc_sha256)
include_directories(.)
set(CMAKE_CXX_STANDARD 20)

add_library(btc_sha256 STATIC crypto/sha256.cpp)
target_compile_definitions(btc_sha256 PRIVATE DISABLE_OPTIMIZED_SHA256=1)