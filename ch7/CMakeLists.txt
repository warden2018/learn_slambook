cmake_minimum_required(VERSION 3.0)
project(front_end)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3 ${SSE_FLAGS} -msse4")

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/include/eigen3")

add_executable(feature_extraction Orb_cv.cpp)
add_executable(feature_extraction_self Orb_self.cpp)
add_executable(pose_estimation_2d_2d pose_estimation_2d2d.cpp)
add_executable(triangulation triangulation.cpp)
add_executable(triangulation_org triangulation_org.cpp)
add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
add_executable(pose_estimation_3d2d_org pose_estimation_3d2d_org.cpp)

target_link_libraries(feature_extraction ${OpenCV_LIBS})
target_link_libraries(feature_extraction_self ${OpenCV_LIBS})
target_link_libraries(pose_estimation_2d_2d ${OpenCV_LIBS})
target_link_libraries(triangulation ${OpenCV_LIBS})
target_link_libraries(triangulation_org ${OpenCV_LIBS})
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} ${Sophus_LIBRARIES} g2o_core g2o_stuff)
target_link_libraries(pose_estimation_3d2d_org ${OpenCV_LIBS} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
