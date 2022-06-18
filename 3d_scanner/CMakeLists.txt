#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealSensePCLExample)

find_package(PCL)
find_package(VTK)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#set(CMAKE_SKIP_BUILD_RPATH FALSE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")


add_executable(rs-pcl rs-pcl.cpp)
set_property(TARGET rs-pcl PROPERTY CXX_STANDARD 14)
include_directories(/usr/local/include /usr/include /usr/local/Cellar/librealsense/2.50.0/include /usr/local/Cellar/pcl/1.12.1_2/include/pcl-1.12 /usr/local/Cellar/eigen/3.4.0_1/include/eigen3 ${VTK_INCLUDE_DIRS})
target_link_directories(rs-pcl PUBLIC /usr/local/lib /usr/local/Cellar/librealsense/2.50.0/lib ${PCL_LIBRARY_DIRS})
target_link_libraries(rs-pcl realsense2 glfw "-framework OpenGL" ${PCL_LIBRARIES} ${VTK_LIBRARIES}) # -lpcl_io -lpcl_common -lpcl_features -lpcl_kdtree -lpcl_visualization -lpcl_surface -lpcl_filters -lpcl_apps -lpcl_cc_tool_euclidean_clustering -lpcl_cc_tool_fpfh_estimation -lpcl_cc_tool_normal_estimation -lpcl_cc_tool_organized_segmentation -lpcl_cc_tool_sanitize_cloud -lpcl_cc_tool_statistical_outlier_removal -lpcl_cc_tool_supervoxels -lpcl_cc_tool_voxel_grid_downsample -lpcl_common -lpcl_features -lpcl_filters -lpcl_io -lpcl_io_ply -lpcl_kdtree -lpcl_keypoints -lpcl_ml -lpcl_octree -lpcl_outofcore -lpcl_people -lpcl_recognition -lpcl_registration -lpcl_sample_consensus -lpcl_search -lpcl_segmentation -lpcl_simulation -lpcl_simulation_io -lpcl_stereo -lpcl_surface -lpcl_tracking -lpcl_visualization)

vtk_module_autoinit(
TARGETS rs-pcl
MODULES ${VTK_LIBRARIES}
)

foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MT")
endforeach(flag_var)

install(
    TARGETS

    rs-pcl

    RUNTIME DESTINATION
    ${CMAKE_INSTALL_PREFIX}/bin
)
