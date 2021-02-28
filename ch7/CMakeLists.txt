cmake_minimum_required(VERSION 3.0)
project(front_end)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("usr/include/eigen3")

add_executable(feature_extraction Orb_cv.cpp)
add_executable(feature_extraction_self Orb_self.cpp)
add_executable(pose_estimation_2d_2d pose_estimation_2d2d.cpp)


target_link_libraries(feature_extraction ${OpenCV_LIBS})
target_link_libraries(feature_extraction_self ${OpenCV_LIBS})
target_link_libraries(pose_estimation_2d_2d ${OpenCV_LIBS})
