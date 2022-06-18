
#include <librealsense2/rs.hpp>
#include "./example.hpp"

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>

#include <pcl/common/io.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>


#define VDOT(a, b) ((a)[0]*(b)[0] + (a)[1]*(b)[1] + (a)[2]*(b)[2])


/* helper brevity */
using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using pcl_rgbptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;


/* for managing rotation of pointcloud view */
struct state {
  state() : yaw(0.0), pitch(0.0), last_x(0.0), last_y(0.0),
            offset_x(0.0f), offset_y(0.0f),
            ml(false), draw0(false), draw1(false), draw2(false), draw3(false) {}
  double yaw, pitch, last_x, last_y;
  float offset_x, offset_y;
  bool ml, draw0, draw1, draw2, draw3;
};


static float3 colors[] { { 0.8f, 0.1f, 0.3f },
                         { 0.3f, 0.8f, 0.1f },
                         { 0.1f, 0.3f, 0.8f },
                         { 0.8f, 0.3f, 0.1f },
                         { 0.1f, 0.8f, 0.3f },
                         { 0.3f, 0.1f, 0.8f },
                         { 0.9f, 0.1f, 0.5f },
                         { 0.5f, 0.9f, 0.1f },
                         { 0.1f, 0.5f, 0.9f },
                         { 0.9f, 0.5f, 0.1f },
                         { 0.1f, 0.9f, 0.5f },
                         { 0.5f, 0.1f, 0.9f }
};


// Handles all the OpenGL calls needed to display the point cloud
static void
draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points) {
  // OpenGL commands that prep screen for the pointcloud
  glPopMatrix();
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  float width = app.width(), height = app.height();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  gluPerspective(60, width / height, 0.01, 10.0);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

  glTranslatef(0, 0, +0.5 + app_state.offset_y*0.05);
  glRotated(app_state.pitch, 1, 0, 0);
  glRotated(app_state.yaw, 0, 1, 0);
  glTranslatef(0, 0, -0.5);

  glPointSize(width / 640 * 3);
  glEnable(GL_TEXTURE_2D);

  int color = 0;

  for (auto&& pc : points) {
    auto c = colors[(color++) % (sizeof(colors) / sizeof(float3))];

    glBegin(GL_POINTS);
    glColor3f(c.x, c.y, c.z);

    std::cout << "drawing " << pc->points.size() << " xyz points" << std::endl;

    for (int i = 0; i < pc->points.size(); i++) {
      auto&& p = pc->points[i];
      if (p.z) {
        glVertex3f(p.x, p.y, p.z);
      }
    }

    glEnd();
  }

  // OpenGL cleanup
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glPopAttrib();
  glPushMatrix();
}


static void
draw_pointcloud(window& app, state& app_state, const std::vector<pcl_rgbptr>& points) {
  // OpenGL commands that prep screen for the pointcloud
  glPopMatrix();
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  float width = app.width(), height = app.height();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  gluPerspective(60, width / height, 0.01, 10.0);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

  glTranslatef(0, 0, +0.5 + app_state.offset_y*0.05);
  glRotated(app_state.pitch, 1, 0, 0);
  glRotated(app_state.yaw, 0, 1, 0);
  glTranslatef(0, 0, -0.5);

  glPointSize(width / 640 * 1);
  glEnable(GL_TEXTURE_2D);

  int color = 2;

  for (auto&& pc : points) {
    auto c = colors[(color++) % (sizeof(colors) / sizeof(float3))];

    std::cout << "drawing " << pc->points.size() << " rgbxyz points" << std::endl;
    glBegin(GL_POINTS);
    //    glColor3f(c.x, c.y, c.z);
    glColor3f(0.0, 0.0, 1.0);

    for (int i = 0; i < pc->points.size(); i++) {
      auto&& p = pc->points[i];
      if (p.z) {
        glVertex3f(p.x, p.y, p.z);
      }
    }

    glEnd();
  }

  // OpenGL cleanup
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glPopAttrib();
  glPushMatrix();
}


// Registers the state variable and callbacks to allow mouse control of the pointcloud
static void
register_glfw_callbacks(window& app, state& app_state) {
  app.on_left_mouse = [&](bool pressed) {
    app_state.ml = pressed;
  };

  app.on_mouse_scroll = [&](double xoffset, double yoffset) {
    app_state.offset_x += static_cast<float>(xoffset);
    app_state.offset_y += static_cast<float>(yoffset);
  };

  app.on_mouse_move = [&](double x, double y) {
    if (app_state.ml) {
      app_state.yaw -= (x - app_state.last_x);
      app_state.yaw = std::max(app_state.yaw, -120.0);
      app_state.yaw = std::min(app_state.yaw, +120.0);
      app_state.pitch += (y - app_state.last_y);
      app_state.pitch = std::max(app_state.pitch, -80.0);
      app_state.pitch = std::min(app_state.pitch, +80.0);
    }
    app_state.last_x = x;
    app_state.last_y = y;
  };

  app.on_key_release = [&](int key) {
    if (key == GLFW_KEY_ESCAPE || key == 'Q') {
      glfwSetWindowShouldClose(app, GLFW_TRUE);
    } else if (key == 'R' /* reset */) {
      app_state.yaw = app_state.pitch = 0; app_state.offset_x = app_state.offset_y = 0.0;
    } else if (key == '0') {
      app_state.draw0 = !app_state.draw0;
    } else if (key == '1') {
      app_state.draw1 = !app_state.draw1;
    } else if (key == '2') {
      app_state.draw2 = !app_state.draw2;
    } else if (key == '3') {
      app_state.draw3 = !app_state.draw3;
    } else {
      printf("key pressed == [%d]\n", key);
    }
  };
}


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


#define USING_GLFW
//#define USING_PCLVIS


int
main(int argc, char * argv[]) try {

#ifdef USING_GLFW
  // use glfw windowing
  window app(1280, 720, "3D Scanner");
  state app_state;
  register_glfw_callbacks(app, app_state);
#endif

#ifdef USING_PCLVIS
  pcl::visualization::PCLVisualizer app("Segmentation");
  app.setBackgroundColor(1, 1, 1);
  app.initCameraParameters ();
  app.setWindowName("Segmentation");
  //pcl::visualization::CloudViewer app("Segmentation");
#endif

  // Declare pointcloud object, for calculating pointclouds and texture mappings
  rs2::pointcloud pc;
  // We want the points object to be persistent so we can display the last cloud when a frame drops
  rs2::points rs_points;

  // Declare RealSense pipeline, encapsulating the actual device and sensors
  rs2::pipeline pipe;
  // Start streaming with default recommended configuration
  rs2::pipeline_profile pipeProfile = pipe.start();


  // We drop a few frames in order for auto-exposure to settle
  auto frames = pipe.wait_for_frames();
  for (int i = 0; i < 20; i++) {
    frames = pipe.wait_for_frames();
  }

  // get first frame to initialize center
  auto depth = frames.get_depth_frame();

  std::cout << "width is " << depth.get_width() << " and height is " << depth.get_height() << std::endl;

  double dist_to_center = depth.get_distance(depth.get_width() / 2, depth.get_height() / 2);
  std::cout << "distance to center: " << dist_to_center << std::endl;

  rs_points = pc.calculate(depth);


  // auto itx = pipeProfile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();

  while (
#ifdef USING_GLFW
         app
#else
#  ifdef USING_PCLVIS
         !app.wasStopped()
#  else
         1
#  endif
#endif
         ) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr object_points (new pcl::PointCloud<pcl::PointXYZ>);

    // Wait for the next set of frames from the camera
    frames = pipe.wait_for_frames();
    depth = frames.get_depth_frame();

    // Generate the pointcloud and texture mappings
    rs_points = pc.calculate(depth);
    auto all_points = points_to_pcl(rs_points);

    dist_to_center = depth.get_distance(depth.get_width() / 2, depth.get_height() / 2);
    // std::cout << "distance to center: " << dist_to_center << std::endl;

    auto center = all_points->at(all_points->width / 2, all_points->height / 2);
    std::cout << "center point is (" << double(center.x) << ", " << double(center.y) << ", " << double(center.z) << ")" << std::endl;


    /* filter out anything too close or too far */
    /* approximating a half-meter cube of interest */
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(all_points);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.15, 0.65);
    pass.filter(*object_points);
    pass.setInputCloud(object_points);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-0.25, 0.25);
    pass.filter(*object_points);
    pass.setInputCloud(object_points);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.25, 0.25);
    pass.filter(*object_points);


    /* reduce to a voxel grid in order to remain interactive */
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(object_points);
    vg.setLeafSize(0.005f, 0.005f, 0.005f);
    // vg.setMinimumPointsNumberPerVoxel(2);
    vg.filter(*object_points);


    /* filter out noise */
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(object_points);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*object_points);


#if 0
    /* remove exterior edge points */
    /* NFG, looses too much of the foreground */
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setInputCloud(object_points);
    outrem.setRadiusSearch(0.1);
    outrem.setMinNeighborsInRadius(2);
    //      outrem.setKeepOrganized(true);
    outrem.filter(*object_points);
#endif


    /* make sure we haven't filtered out everything */
    if (object_points->size() == 0)
      continue;

    /* identify the ground plane */
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setInputCloud(object_points);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.003); /* +-3mm tol at 500mm */
    seg.segment(*inliers, *coefficients);

    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_points (new pcl::PointCloud<pcl::PointXYZ>);

    std::cout << "Model coefficients: " << coefficients->values[0] << " "
              << coefficients->values[1] << " "
              << coefficients->values[2] << " "
              << coefficients->values[3] << std::endl;


    /* check if the plane is principally horizontal w.r.t. the Y axis
     * (i.e., it's Y-up in the XZ-plane)
     */
    double xup[3] = {1.0, 0.0, 0.0};
    double yup[3] = {0.0, 1.0, 0.0};
    double zup[3] = {0.0, 0.0, 1.0};
    double xdot = VDOT(xup, coefficients->values) / coefficients->values[3];
    double ydot = VDOT(yup, coefficients->values) / coefficients->values[3];
    double zdot = VDOT(zup, coefficients->values) / coefficients->values[3];
    // std::cout << "XDOT=" << xdot << " YDOT=" << ydot << " ZDOT=" << zdot << std::endl;

    if (ydot < xdot && ydot < zdot) {

      /* filter out the horizontal plane points */
      pcl::copyPointCloud(*object_points, *inliers, *ground_points);

      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(object_points);
      extract.setIndices(inliers);
      extract.setNegative(true);
      extract.filter(*object_points);


      /* filter out anything remaining below the ground plane */
      for (pcl::PointCloud<pcl::PointXYZ>::iterator it = object_points->begin(); it != object_points->end();) {
        static int subset = 0;
        Eigen::Vector4f plane;
        plane[0] = coefficients->values[0];
        plane[1] = coefficients->values[1];
        plane[2] = coefficients->values[2];
        plane[3] = coefficients->values[3];

        double d = pcl::pointToPlaneDistanceSigned(*it, plane);
        if (subset++ % 1000000 == 0)
          std::cout << "dist to plane: " << d << std::endl;

        if (d < 0.002) { /* trim 2mm above the plane */
          it = object_points->erase(it);
        } else {
          ++it;
        }
      }
    }


    /* use region growing segmentation to find the foreground cluster based on normals */
    pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(object_points);
    //    normal_estimator.setKSearch(25); /* use local cluster to estimate normal */
    normal_estimator.setRadiusSearch(0.05); /* use 5mm local area to estimate normal */
    normal_estimator.compute(*normals);

    pcl::IndicesPtr indices (new std::vector <int>);
    pcl::removeNaNFromPointCloud (*object_points, *indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setInputCloud(object_points);
    reg.setInputNormals(normals);
    reg.setIndices(indices);

    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(5);
    //    reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
    //    reg.setCurvatureThreshold(1.0);
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(100000);

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);
    std::cout << "#Clusters is " << clusters.size();
    if (clusters.size() > 0)
      std::cout << " and cluster[0] size is " << clusters[0].indices.size() << std::endl;
    else
      std::cout << std::endl;

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr region_points = reg.getColoredCloud();


#if 0
    /* Attempted MinCut to extract foreground, INEFFECTIVE */
    /* super-slow at default res, but interactive w/ voxel grid */

    pcl::IndicesPtr indices(new std::vector <int>);
    pcl::removeNaNFromPointCloud(*object_points, *indices);

    pcl::MinCutSegmentation<pcl::PointXYZ> mcseg;
    mcseg.setInputCloud(object_points);
    mcseg.setIndices(indices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr foreground_points(new pcl::PointCloud<pcl::PointXYZ>());
    foreground_points->points.push_back(center);
    mcseg.setForegroundPoints(foreground_points);

    mcseg.setSigma(.025); /* 200mm connected component size */
    mcseg.setRadius(.01); /* not bigger than half the scan volume +-125mm */
    mcseg.setNumberOfNeighbours(1); /* half the 3x3 */
    mcseg.setSourceWeight(.01);

    std::vector <pcl::PointIndices> clusters;
    mcseg.extract(clusters);

    std::cout << "Maximum flow is " << mcseg.getMaxFlow() << std::endl;

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr region_points = mcseg.getColoredCloud();

#  ifdef USING_PCLVIS
    static int added = 0;
    if (added++)
      app.updatePointCloud(region_points);
    else
      app.addPointCloud<pcl::PointXYZRGB>(region_points);
#  endif
#endif



#if 0
    /* attempted RANSAC to extract ground plane, INEFFECITVE */
    std::vector<int> inliers;
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr planar (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (object_points));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(planar);
    ransac.setDistanceThreshold(1);
    ransac.computeModel();
    ransac.getInliers(inliers);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_points (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*object_points, inliers, *ground_points);
#endif


    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(1., 1., 1., 1);
    // glClearColor(192. / 255, 192. / 255, 192. / 255, 0.5);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPointSize(3.0f);


#ifdef USING_GLFW
    std::vector<pcl_ptr> layers;
    if (app_state.draw0)
      layers.push_back(all_points);
    if (app_state.draw1)
      layers.push_back(object_points);
    if (app_state.draw2)
      layers.push_back(ground_points);
    draw_pointcloud(app, app_state, layers);

    //#else

    std::vector<pcl_rgbptr> layers2;
    if (app_state.draw3)
      layers2.push_back(region_points);

      draw_pointcloud(app, app_state, layers2);
#else
#  ifdef USING_PCLVIS
    //app.addPointCloud(region_points);//, pt_handler, "region_points");
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pt_handler (region_points, 0, 0, 0);
    app.spinOnce();
#  endif
#endif

  }

  return EXIT_SUCCESS;

} catch (const rs2::error & e) {

  std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
  return EXIT_FAILURE;

} catch (const std::exception & e) {

  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;

}

