cmake_minimum_required(VERSION 3.0)
project(stereoVision)
set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(stereoVision StereoVision.cpp)

target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})