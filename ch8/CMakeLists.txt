cmake_minimum_required(VERSION 3.0)
project(front_end_2)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 ${SSE_FLAGS} -msse4")

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
SET(gflags_DIR "/usr/local/lib/cmake/gflags/")
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(gflags REQUIRED)
message("OpenCV version: " ${OpenCV_VERSION})
# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )
# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/include/eigen3")

add_executable(optical_flow optical_flow.cpp)
add_executable(optical_flow_org optical_flow_org.cpp)
add_executable(direct_method direct_method.cpp common/directMethod_flags.h common/directMethod_flags.cpp)
add_executable(direct_method_org direct_method_org.cpp)

target_link_libraries(optical_flow ${OpenCV_LIBS} ${Sophus_LIBRARIES} g2o_core g2o_stuff)
target_link_libraries(optical_flow_org ${OpenCV_LIBS} ${Sophus_LIBRARIES} g2o_core g2o_stuff)
target_link_libraries(direct_method ${OpenCV_LIBS} ${Sophus_LIBRARIES} gflags ${Pangolin_LIBRARIES})
target_link_libraries(direct_method_org ${OpenCV_LIBS} ${Sophus_LIBRARIES} gflags ${Pangolin_LIBRARIES})