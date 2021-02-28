cmake_minimum_required (VERSION 2.8)
project(UseEigen)

include_directories("usr/include/eigen3")

add_executable(useEigen eigenMatrix.cpp)