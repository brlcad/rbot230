#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(3d-scanner)

find_package(PCL 1.2 REQUIRED)
find_package(VTK REQUIRED)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#set(CMAKE_SKIP_BUILD_RPATH FALSE)

# the RPATH to be used when installing, but only if it's not a system directory
#list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#if("${isSystemDir}" STREQUAL "-1")
#    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#endif("${isSystemDir}" STREQUAL "-1")


add_executable(3d-scanner 3d-scanner.cpp)
set_property(TARGET 3d-scanner PROPERTY CXX_STANDARD 14)
include_directories(/usr/local/include /usr/include /usr/local/Cellar/librealsense/2.50.0/include ${PCL_INCLUDE_DIRS} ${VTK_USE_FILE} ${VTK_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_directories(3d-scanner PUBLIC /usr/local/lib /usr/local/Cellar/librealsense/2.50.0/lib ${PCL_LIBRARY_DIRS})
target_link_libraries(3d-scanner  ${PCL_LIBRARIES} ${VTK_LIBRARIES} realsense2 glfw "-framework OpenGL")

vtk_module_autoinit(
TARGETS 3d-scanner
MODULES ${VTK_LIBRARIES}
)

install(
    TARGETS

    3d-scanner

    RUNTIME DESTINATION
    ${CMAKE_INSTALL_PREFIX}/bin
)
