#include <librealsense2/rs.hpp>

#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter_indices.h> // for pcl::removeNaNFromPointCloud
#include <pcl/segmentation/min_cut_segmentation.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/visualization/pcl_visualizer.h>


using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

static pcl_ptr
points_to_pcl(const rs2::points& points) {
  pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  auto sp = points.get_profile().as<rs2::video_stream_profile>();
  cloud->width = sp.width();
  cloud->height = sp.height();
  cloud->is_dense = false;
  cloud->points.resize(points.size());
  auto ptr = points.get_vertices();
  for (auto& p : cloud->points) {
    p.x = ptr->x;
    p.y = ptr->y;
    p.z = ptr->z;
    ptr++;
  }

  return cloud;
}


int main ()
{
  rs2::pointcloud pc;
  rs2::pipeline pipe;
  rs2::pipeline_profile pipeProfile = pipe.start();
  auto frames = pipe.wait_for_frames();
  auto depth = frames.get_depth_frame();
  rs2::points rs_points = pc.calculate(depth);
  auto cloud = points_to_pcl(rs_points);

  //  pcl::IndicesPtr indices (new std::vector <int>);
  //  pcl::removeNaNFromPointCloud(*cloud, *indices);

  pcl::PointCloud<pcl::PointXYZ>::Ptr object_points (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(0.05f, 0.05f, 0.05f);
  // vg.setMinimumPointsNumberPerVoxel(2);
  vg.filter(*object_points);

#if 0
  pcl::MinCutSegmentation<pcl::PointXYZ> seg;
  seg.setInputCloud (object_points);
  //  seg.setIndices (indices);

  auto center = object_points->at(object_points->width / 2, object_points->height / 2);
  std::cout << "center point is (" << double(center.x) << ", " << double(center.y) << ", " << double(center.z) << ")" << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr foreground_points(new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointXYZ point;
  point.x = center.x;
  point.y = center.y;
  point.z = center.z;
  foreground_points->points.push_back(point);
  seg.setForegroundPoints (foreground_points);

  seg.setSigma (0.25);
  seg.setRadius (3.0433856);
  seg.setNumberOfNeighbours (14);
  seg.setSourceWeight (0.8);

  std::vector <pcl::PointIndices> clusters;
  seg.extract (clusters);

  std::cout << "Maximum flow is " << seg.getMaxFlow () << std::endl;

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = seg.getColoredCloud ();
#endif


  pcl::visualization::PCLVisualizer app("Segmentation");
  app.setBackgroundColor(0,0,0);
  app.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  app.addCoordinateSystem (1.0);
  app.initCameraParameters ();
  app.setWindowName("Segmentation");
  //app.addPointCloud(colored_cloud);
  app.addPointCloud(object_points);
  while (!app.wasStopped ())
  {
    frames = pipe.wait_for_frames();
    depth = frames.get_depth_frame();
    rs_points = pc.calculate(depth);
    cloud = points_to_pcl(rs_points);

    pcl::PointCloud<pcl::PointXYZ>::Ptr object_points (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    // vg.setMinimumPointsNumberPerVoxel(2);
    vg.filter(*object_points);


    //app.updatePointCloud(colored_cloud);
    app.updatePointCloud(object_points);

    std::cout << "cloud has " << object_points->size() << " points" << std::endl;

    app.spinOnce();
  }

  return (0);
}
