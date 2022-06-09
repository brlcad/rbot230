#include <memory>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>


static void
setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}



int main () {
  // Generate the data
  FILE *xyz = fopen("../bunny.xyz", "r");
  if (!xyz)
    return 1;

  pcl::PointCloud<pcl::PointXYZ> pointCloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudPtr(&pointCloud);

#define BUFSIZE 2048
  char buf[BUFSIZE] = {0};
  while (fgets(buf, BUFSIZE, xyz)) {
    double x, y, z;
    sscanf(buf, "%lf %lf %lf", &x, &y, &z);

    pcl::PointXYZ point;
    point.x = x;
    point.y = y;
    point.z = z;
    pointCloud.push_back(point);
  }

#if 0
  for (float y=-0.5f; y<=0.5f; y+=0.01f) {
    for (float z=-0.5f; z<=0.5f; z+=0.01f) {
      pcl::PointXYZ point;
      point.x = 2.0f - y;
      point.y = y;
      point.z = z;
      pointCloud.push_back(point);
    }
  }
  #endif

  pointCloud.width = pointCloud.size();
  pointCloud.height = 1;


  // We now want to create a range image from the above point cloud,
  // with a 1deg angular resolution
  float angularResolution = (float) (  1.0f * (M_PI/180.0f));  //   1.0 degree in radians
  float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
  float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
  pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
  float noiseLevel=0.00;
  float minRange = 0.0f;
  int borderSize = 1;

  //  std::shared_ptr<pcl::RangeImage> rangeImage;
  pcl::RangeImage::Ptr range_image_ptr(new pcl::RangeImage);
  pcl::RangeImage& rangeImage = *range_image_ptr;
  rangeImage.createFromPointCloud(pointCloud, angularResolution,
                                  maxAngleWidth, maxAngleHeight,
                                  sensorPose, coordinate_frame,
                                  noiseLevel, minRange, borderSize);

  std::cout << rangeImage << "\n";

  //  pcl::RangeImage::Ptr range_image_ptr = rangeImage;
  pcl::visualization::PCLVisualizer viewer ("3D Viewer");
  viewer.setBackgroundColor (1, 1, 1);

  // Filter input points into one point per voxel grid as a simplistic
  // means for point decimation.
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(pointCloudPtr);
  vg.setLeafSize(0.01f, 0.01f, 0.01f);
  vg.filter(*pointCloudPtr);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> vg_handler (pointCloudPtr, 0, 0, 0);
  viewer.addPointCloud (pointCloudPtr, vg_handler, "voxels");

  //  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 0, 0, 0);
  //  viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "image");

  //viewer.addCoordinateSystem (1.0f, "global");
  //PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 150, 150, 150);
  //viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
  viewer.initCameraParameters ();
  setViewerPose(viewer, rangeImage.getTransformationToWorldSystem ());

  // --------------------------
  // -----Show range image-----
  // --------------------------
  pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");
  range_image_widget.showRangeImage(rangeImage);

  //--------------------
  // -----Main loop-----
  //--------------------
#if 1
  Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
  while (!viewer.wasStopped ()) {
    range_image_widget.spinOnce ();
    viewer.spinOnce ();
    pcl_sleep (0.01);

  }
#endif
  pcl_sleep (10);

  return 0;
}
