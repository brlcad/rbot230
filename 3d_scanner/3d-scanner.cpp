
#include <librealsense2/rs.hpp>
#include "./example.hpp"

#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/features/normal_3d.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>

#include <pcl/common/io.h>
#include <pcl/common/distances.h>

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

  if (points.size() == 0)
    return;

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

    if (!pc || pc->points.size() == 0)
      continue;

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

  if (points.size() == 0)
    return;

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

  glPointSize(width / 640 * 2);
  glEnable(GL_TEXTURE_2D);

  int color = 2;

  for (auto&& pc : points) {

    if (!pc || pc->points.size() == 0)
      continue;

    std::cout << "drawing " << pc->points.size() << " rgbxyz points" << std::endl;
    glBegin(GL_POINTS);
    //    glColor3f(c.x, c.y, c.z);

    for (int i = 0; i < pc->points.size(); i++) {
      auto&& p = pc->points[i];

      //std::cout << "rgb is " << (int)p.r << "," << (int)p.g << "," << (int)p.b << std::endl;

      /* convert to hsv so we can display light colors darkly */
      pcl::PointXYZHSV h;
      pcl::PointXYZRGBtoXYZHSV(p, h);
      // std::cout << "hsv is " << (int)h.h << "," << (float)h.s << "," << (float)h.v << std::endl;
      if (1 /*h.v > 0.75*/) {
        h.v *= 0.01;
        pcl::PointXYZHSVtoXYZRGB(h, p);
        // std::cout << "rgb AFTER is " << (int)p.r << "," << (int)p.g << "," << (int)p.b << std::endl;
      }

      glColor3f(p.r, p.g, p.b);
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
draw_grid(window& app, state& app_state, const std::vector<pcl::PointXYZ>&points) {

  if (points.size() == 0)
    return;

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

  glPointSize(width / 640 * 4);
  glEnable(GL_TEXTURE_2D);

  glBegin(GL_POINTS);
  glColor3f(0.0, 0.0, 0.0);
  for (auto&& p : points) {
    glVertex3f(p.x, p.y, p.z);
  }
  glEnd();

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


static pcl::PointCloud<pcl::PointXYZ>::Ptr
pointsNearPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr pnts, pcl::PointXYZ point, const double radius) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr newPnts(new pcl::PointCloud<pcl::PointXYZ>);
  float distance;
  for (int i = 0; i < pnts->size(); ++i) {
    distance = pcl::euclideanDistance(pnts->at(i), point);
    if (distance <= radius) {
      newPnts->push_back(pnts->at(i));
    }
  }
  return newPnts;
}


#if 0
pcl::PointCloud<pcl::PointXYZ>::Ptr
operator+=(pcl::PointCloud<pcl::PointXYZ>::Ptr pnts) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr newPnts(new pcl::PointCloud<pcl::PointXYZ>);
  float distance;
  for (int i = 0; i < pnts->size(); ++i) {
    distance = pcl::euclideanDistance(pnts->at(i), point);
    if (distance <= radius) {
      newPnts->push_back(pnts->at(i));
    }
  }
  return newPnts;
}
#endif


static bool
compare_point(pcl::PointXYZ p1, pcl::PointXYZ p2) {
  if (p1.x != p2.x)
    return p1.x > p2.x;
  else if (p1.y != p2.y)
    return  p1.y > p2.y;
  else
    return p1.z > p2.z;
}


static bool
is_same_point(pcl::PointXYZ p1, pcl::PointXYZ p2) {
  /* exact same point, no fuzz */
  if (p1.x == p2.x && p1.y == p2.y && p1.z == p2.z)
    return true;
  return false;
}


static Eigen::Vector4f
get_ground_plane(pcl_ptr points, pcl_ptr ground, pcl::PointIndices::Ptr inliers) {
    /* identify the ground plane */
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    //    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setInputCloud(points);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.003); /* +-3mm tol at 500mm */
    seg.segment(*inliers, *coefficients);

    /*
      std::cout << "Model coefficients: " << coefficients->values[0] << " "
              << coefficients->values[1] << " "
              << coefficients->values[2] << " "
              << coefficients->values[3] << std::endl;
    */


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

    Eigen::Vector4f plane;
    plane[0] = coefficients->values[0];
    plane[1] = coefficients->values[1];
    plane[2] = coefficients->values[2];
    plane[3] = coefficients->values[3];

    if (ydot < xdot && ydot < zdot) {
      pcl::copyPointCloud(*points, *inliers, *ground);

      return plane;
    }

    return Eigen::Vector4f::Zero();
}


// FIXME: should really take the plane equation instead of drawing a
// plane through the bounding volume assuming ymin/ymax tilts towards
// the camera.
static void
draw_plane(window& app, state& app_state, const pcl_ptr& points) {

  if (points->size() == 0)
    return;

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

  double xmin = 1000000, ymin = 1000000, zmin = 1000000, xmax = -1000000, ymax = -1000000, zmax = -1000000;

  for (auto&& p : *points) {
    if (p.x < xmin)
      xmin = p.x;
    if (p.x > xmax)
      xmax = p.x;
    if (p.y < ymin)
      ymin = p.y;
    if (p.y > ymax)
      ymax = p.y;
    if (p.z < zmin)
      zmin = p.z;
    if (p.z > zmax)
      zmax = p.z;
  }

  // std::cout << "min/max = " << xmin << "," << ymin << "," << zmin << " to " << xmax << "," << ymax << "," << zmax << std::endl;

  glColor3f(0.0, 0.3, 0.0);
  glBegin(GL_POLYGON);
  glVertex3f(xmin, ymax, zmin);
  glVertex3f(xmax, ymax, zmin);
  glVertex3f(xmax, ymin, zmax);
  glVertex3f(xmin, ymin, zmax);
  glEnd();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glPopAttrib();
  glPushMatrix();
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
  app.initCameraParameters();
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

    pcl_ptr object_points(new pcl::PointCloud<pcl::PointXYZ>);

    // Wait for the next set of frames from the camera
    frames = pipe.wait_for_frames();
    depth = frames.get_depth_frame();

    // Generate the pointcloud and texture mappings
    rs_points = pc.calculate(depth);
    auto all_points = points_to_pcl(rs_points);

    dist_to_center = depth.get_distance(depth.get_width() / 2, depth.get_height() / 2);
    // std::cout << "distance to center: " << dist_to_center << std::endl;


    /*
     * We calculate a sampling grid that is a 3x3 pattern centered in
     * the view with 10% coverage horizontally and vertically.
     *
     *
     *    tl---tm---tr
     *    |     |    |
     *    ml-center-mr
     *    |     |    |
     *    bl---bm---br
     *
     */
    double center_x = all_points->width / 2.0;
    double center_y = all_points->height / 2.0;
    double vertical = center_y / 5.0;
    double horizontal = center_x / 5.0;

    auto tl = all_points->at(center_x - horizontal, center_y - vertical);
    auto tm = all_points->at(center_x, center_y - vertical);
    auto tr = all_points->at(center_x + horizontal, center_y - vertical);

    auto ml = all_points->at(center_x - horizontal, center_y);
    auto center = all_points->at(center_x, center_y);
    auto mr = all_points->at(center_x + horizontal, center_y);

    auto bl = all_points->at(center_x - horizontal, center_y + vertical);
    auto bm = all_points->at(center_x, center_y + vertical);
    auto br = all_points->at(center_x + horizontal, center_y + vertical);

    // std::cout << "tl point is " << tl << std::endl;
    std::cout << "center point is " << center << std::endl;
    // std::cout << "br point is " << br << std::endl;

    std::vector<pcl::PointXYZ> grid;
    grid.push_back(tl);
    grid.push_back(tm);
    grid.push_back(tr);
    grid.push_back(ml);
    grid.push_back(center);
    grid.push_back(mr);
    grid.push_back(bl);
    grid.push_back(bm);
    grid.push_back(br);


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


    /* filter out noise, points not within a standard deviation */
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(object_points);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*object_points);


    /* make sure we haven't filtered out everything */
    if (object_points->size() == 0)
      continue;


    /* compute a ground plane that is principally horizontal with
     * respect to the XZ plane (+Y-up).
     */
    pcl_ptr ground_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    Eigen::Vector4f plane = get_ground_plane(object_points, ground_points, inliers);

    if (plane != Eigen::Vector4f::Zero()) {
#if 0
      /* filter out the horizontal plane points */
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(object_points);
      extract.setIndices(inliers);
      extract.setNegative(true);
      extract.filter(*object_points);
#endif

      /* filter out anything below the ground plane */
      for (pcl::PointCloud<pcl::PointXYZ>::iterator it = object_points->begin(); it != object_points->end();) {
        static int subset = 0;

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


    /* identify center "focus" points using a 3x3 gaussian
     * neighborhood:
     *
     *    50--100--50
     *    |    |    |
     *   100--200--100
     *    |    |    |
     *    50--100--50
     *
     */
    auto cpnts = pointsNearPoint(object_points, center, 0.2);
    auto tlpnts = pointsNearPoint(object_points, tl, 0.05);
    auto tmpnts = pointsNearPoint(object_points, tm, 0.1);
    auto trpnts = pointsNearPoint(object_points, tr, 0.05);
    auto mlpnts = pointsNearPoint(object_points, ml, 0.1);
    auto mrpnts = pointsNearPoint(object_points, mr, 0.1);
    auto blpnts = pointsNearPoint(object_points, bl, 0.05);
    auto bmpnts = pointsNearPoint(object_points, bm, 0.1);
    auto brpnts = pointsNearPoint(object_points, br, 0.05);

    std::cout << std::setw(4);
    std::cout << "counts: " << tlpnts->size() << " - " << tmpnts->size() << " - " << trpnts->size() << std::endl;
    std::cout << "        " << mlpnts->size() << " - " << cpnts->size() << " - " << mrpnts->size() << std::endl;
    std::cout << "        " << blpnts->size() << " - " << bmpnts->size() << " - " << brpnts->size() << std::endl;

    std::cout << "size before: " << object_points->size() << std::endl;

    object_points = cpnts;
    *object_points += *tlpnts;
    *object_points += *tmpnts;
    *object_points += *trpnts;
    *object_points += *mlpnts;
    *object_points += *mrpnts;
    *object_points += *blpnts;
    *object_points += *bmpnts;
    *object_points += *brpnts;

    std::cout << "size after: " << object_points->size() << std::endl;

    std::sort(object_points->begin(), object_points->end(), compare_point);
    auto unique_end = std::unique(object_points->begin(), object_points->end(), is_same_point);
    object_points->erase(unique_end, object_points->end());

    std::cout << "size after: " << object_points->size() << std::endl;


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


    /* filter out any disconnected outlier points, anything not within
     * a threshold distance of other points
     */
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> filter;
    filter.setInputCloud(object_points);
    filter.setRadiusSearch(0.005); /* 5mm radius */
    filter.setMinNeighborsInRadius(1);
    filter.filter(*object_points);


#if 0
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(object_points);
    double radius = .25; /* filter within a 250mm radius */
    std::vector<int> nearby; // index of surrounding points
    std::vector<float> nearbyDistances; // distance to surrounding points
    if (kdtree.radiusSearch(center, radius, nearby, nearbyDistances) > 0) {
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices(nearby));
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(object_points);
      extract.setIndices(inliers);
      extract.setNegative(false);
      extract.filter(*object_points);
    }
#endif


    /* use region growing segmentation to find background clusters based on normals */
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(object_points);
    // normal_estimator.setKSearch(25); /* use local cluster to estimate normal */
    normal_estimator.setRadiusSearch(0.01); /* use 10mm local area to estimate normal */
    normal_estimator.compute(*normals);

    pcl::IndicesPtr indices(new std::vector <int>);
    pcl::removeNaNFromPointCloud(*object_points, *indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setInputCloud(object_points);
    reg.setInputNormals(normals);
    reg.setIndices(indices);

    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(10);
    reg.setSmoothnessThreshold(6.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(100000);

    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);
    pcl_rgbptr region_points = reg.getColoredCloud();

    std::cout << "#Clusters is " << clusters.size();
    if (clusters.size() > 0)
      std::cout << " and cluster[0] size is " << clusters[0].indices.size() << std::endl;
    else
      std::cout << std::endl;

    for (int i = 0; i < clusters.size(); i++) {
      /* TODO: find which cluster has our center point */
    }

#if 0
    /* filter out the clustered background/noise points */
    for (int i = 0; i < clusters.size(); i++) {
      pcl::PointIndices::Ptr outliers(new pcl::PointIndices(clusters[i]));
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(object_points);
      extract.setIndices(outliers);
      extract.setNegative(true);
      extract.filter(*object_points);
    }
#endif


#if 0
    /* Attempted MinCut to extract foreground, INEFFECTIVE */
    /* super-slow at default res, but interactive w/ voxel grid */

    pcl::IndicesPtr indices(new std::vector <int>);
    pcl::removeNaNFromPointCloud(*object_points, *indices);

    pcl::MinCutSegmentation<pcl::PointXYZ> mcseg;
    mcseg.setInputCloud(object_points);
    mcseg.setIndices(indices);

    pcl_ptr foreground_points(new pcl::PointCloud<pcl::PointXYZ>);
    foreground_points->points.push_back(center);
    mcseg.setForegroundPoints(foreground_points);

    mcseg.setSigma(.025); /* 200mm connected component size */
    mcseg.setRadius(.01); /* not bigger than half the scan volume +-125mm */
    mcseg.setNumberOfNeighbours(1); /* half the 3x3 */
    mcseg.setSourceWeight(.01);

    std::vector <pcl::PointIndices> clusters;
    mcseg.extract(clusters);

    std::cout << "Maximum flow is " << mcseg.getMaxFlow() << std::endl;

    pcl_rgbptr region_points = mcseg.getColoredCloud();

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
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr planar(new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (object_points));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(planar);
    ransac.setDistanceThreshold(1);
    ransac.computeModel();
    ransac.getInliers(inliers);
    pcl_ptr ground_points(new pcl::PointCloud<pcl::PointXYZ>);
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
    if (app_state.draw2) {
      draw_plane(app, app_state, ground_points);
      layers.push_back(ground_points);
    }
    draw_pointcloud(app, app_state, layers);

    //#else

    std::vector<pcl_rgbptr> layers2;
    if (app_state.draw3)
      layers2.push_back(region_points);
    draw_pointcloud(app, app_state, layers2);

    draw_grid(app, app_state, grid);
#else
#  ifdef USING_PCLVIS
    //app.addPointCloud(region_points);//, pt_handler, "region_points");
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pt_handler(region_points, 0, 0, 0);
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


