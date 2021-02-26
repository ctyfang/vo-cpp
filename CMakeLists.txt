set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.19.3)
project(vo)

set(OpenCV_DIR /home/carter/repos/opencv-4.4.0/build)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp")

add_executable(vo ${SOURCES})

target_link_libraries(vo ${OpenCV_LIBS} ${Boost_LIBRARIES})