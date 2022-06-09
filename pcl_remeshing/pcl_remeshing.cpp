#include <memory>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/point_types.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/marching_cubes_rbf.h>



static void
setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f(0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f(0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}



int main () {
  pcl::PointCloud<pcl::PointXYZ> pointCloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudPtr(&pointCloud);

  pcl::PLYReader plyreader;
  plyreader.read("/Users/morrison/Desktop/xyzrgb_dragon.ply", *pointCloudPtr);

  pointCloud.width = pointCloud.size();
  pointCloud.height = 1;
  printf("point cloud size is %zu\n", pointCloud.size());

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

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(pointCloudPtr);
  vg.setLeafSize(2.f, 2.f, 2.f);
  vg.filter(*pointCloudPtr);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> vg_handler (pointCloudPtr, 0, 0, 0);
  viewer.addPointCloud (pointCloudPtr, vg_handler, "voxels");

  //  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler (range_image_ptr, 0, 0, 0);
  //  viewer.addPointCloud (range_image_ptr, range_image_color_handler, "range image");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "voxels");

  //viewer.addCoordinateSystem (1.0f, "global");
  //PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 150, 150, 150);
  //viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
  viewer.initCameraParameters();
  //  viewer.resetCameraViewpoint();
  setViewerPose(viewer, rangeImage.getTransformationToWorldSystem ());
  viewer.setCameraPosition(0, 0, 0, 0, 0, 1, 0, 1, 0);

  std::vector<pcl::visualization::Camera> cam;
  viewer.getCameras(cam);

  // cam[0].view[0] = 0;
  // cam[0].view[1] = 0;
  // cam[0].view[2] = 1;
  // cam[0].focal[0] = 0;
  // cam[0].focal[1] = 1;
  // cam[0].focal[2] = 0;

  std::cout << "Cam: " << std::endl
            << " - pos: (" << cam[0].pos[0] << ", " << cam[0].pos[1] << ", " << cam[0].pos[2] << ")" << std::endl
            << " - view: (" << cam[0].view[0] << ", " << cam[0].view[1] << ", " << cam[0].view[2] << ")" << std::endl
            << " - focal: (" << cam[0].focal[0] << ", " << cam[0].focal[1] << ", " << cam[0].focal[2] << ")" << std::endl;

  std::cout << "computing normals" << std::endl;

  // Normal estimation*
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (pointCloudPtr);
  n.setInputCloud (pointCloudPtr);
  n.setSearchMethod (tree);
  n.setKSearch (20);
  n.compute (*normals);

  std::cout << "DONE computing normals" << std::endl;

  // Concatenate the XYZ and normal fields*
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*pointCloudPtr, *normals, *cloud_with_normals);

  // Create search tree*
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud (cloud_with_normals);

  // Initialize objects
  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
  pcl::PolygonMesh triangles;

  // Set the maximum distance between connected points (maximum edge length)
  gp3.setSearchRadius (20);

  // Set typical values for the parameters
  gp3.setMu (2.5);
  gp3.setMaximumNearestNeighbors (100);
  gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
  gp3.setMinimumAngle(M_PI/18); // 10 degrees
  gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
  gp3.setNormalConsistency(false);

  // Get result
  gp3.setInputCloud (cloud_with_normals);
  gp3.setSearchMethod (tree2);
  gp3.reconstruct (triangles);

  std::cout << "greedy count: " << triangles.polygons.size() << std::endl;

  //  viewer.addPolygonMesh(triangles);

#if 0
  pcl::Poisson<pcl::PointNormal> poisson;
  pcl::PolygonMesh triangles2;

  poisson.setSearchMethod(tree2);
  poisson.setDepth(12);
  poisson.setSolverDivide(8);
  poisson.setIsoDivide(8);
  poisson.setPointWeight(10.0f);
  poisson.setInputCloud(cloud_with_normals);
  poisson.reconstruct(triangles2);
  viewer.addPolygonMesh(triangles2);

  std::cout << "poisson count: " << triangles2.polygons.size() << std::endl;
#endif

  pcl::MarchingCubesRBF<pcl::PointNormal> mcrbf(1.0f);
  pcl::PolygonMesh triangles3;
  mcrbf.setInputCloud(cloud_with_normals);
  mcrbf.setSearchMethod(tree2);
  mcrbf.reconstruct(triangles3);
  viewer.addPolygonMesh(triangles3);

  std::cout << "marching cubes rbf count: " << triangles3.polygons.size() << std::endl;

  // --------------------------
  // -----Show range image-----
  // --------------------------
  //  pcl::visualization::RangeImageVisualizer range_image_widget ("Range image");
  //  range_image_widget.showRangeImage(rangeImage);

  //--------------------
  // -----Main loop-----
  //--------------------
#if 1
  Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
  while (!viewer.wasStopped ()) {
    //    range_image_widget.spinOnce ();
    viewer.spinOnce ();
    pcl_sleep (0.01);

  }
#endif
  pcl_sleep (10);

}
