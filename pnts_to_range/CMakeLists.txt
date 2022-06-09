cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(range_image_creation)

find_package(PCL 1.2 REQUIRED)
find_package(VTK REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${VTK_USE_FILE} ${VTK_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} -g -O0 -std=gnu++14)

add_executable (range_image_creation range_image_creation.cpp)
target_link_libraries (range_image_creation ${PCL_LIBRARIES} ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS ${PROJECT_NAME}
  MODULES ${VTK_LIBRARIES}
)
