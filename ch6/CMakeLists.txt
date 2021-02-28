cmake_minimum_required(VERSION 3.0)
project(guassNewton)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("usr/include/eigen3")

add_executable(guassNewton GuassNewton.cpp)
add_executable(ceresCurveFitting ceresCurveFitting.cpp)
add_executable(g2oCurveFitting g2oCurveFitting.cpp)

target_link_libraries(guassNewton ${OpenCV_LIBS})
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
