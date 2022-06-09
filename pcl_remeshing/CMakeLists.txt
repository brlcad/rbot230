cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(pcl_remeshing)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} -g -O0)

add_executable (pcl_remeshing pcl_remeshing.cpp)
target_link_libraries (pcl_remeshing ${PCL_LIBRARIES})
