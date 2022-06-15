#include <librealsense2/rs.hpp>
#include "./example.hpp"

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include <pcl/segmentation/min_cut_segmentation.h>

#if 0
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#endif


// Struct for managing rotation of pointcloud view
struct state {
  state() : yaw(0.0), pitch(0.0), last_x(0.0), last_y(0.0),
            ml(false), offset_x(0.0f), offset_y(0.0f) {}
  double yaw, pitch, last_x, last_y; bool ml; float offset_x, offset_y;
};


using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using pcl_rgbptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;

// Helper functions
void register_glfw_callbacks(window& app, state& app_state);
void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points);
void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_rgbptr>& points);

pcl_ptr points_to_pcl(const rs2::points& points)
{
  pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  auto sp = points.get_profile().as<rs2::video_stream_profile>();
  cloud->width = sp.width();
  cloud->height = sp.height();
  cloud->is_dense = false;
  cloud->points.resize(points.size());
  auto ptr = points.get_vertices();
  for (auto& p : cloud->points)
    {
      p.x = ptr->x;
      p.y = ptr->y;
      p.z = ptr->z;
      ptr++;
    }

  return cloud;
}


float3 colors[] { { 0.8f, 0.1f, 0.3f },
                  { 0.1f, 0.9f, 0.5f },
};


int main(int argc, char * argv[]) try
  {
    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense PCL Pointcloud Example");
    // Construct an object to manage view state
    state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points rs_points;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    rs2::pipeline_profile pipeProfile = pipe.start();

    // get first frame to initialize center
    auto frames = pipe.wait_for_frames();
    auto depth = frames.get_depth_frame();

    std::cout << "width is " << depth.get_width() << " and height is " << depth.get_height() << std::endl;

    double dist_to_center = depth.get_distance(depth.get_width() / 2, depth.get_height() / 2);
    std::cout << "distance to center: " << dist_to_center << std::endl;

    rs_points = pc.calculate(depth);

    // auto itx = pipeProfile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();

    while (app) {

      // Wait for the next set of frames from the camera
      frames = pipe.wait_for_frames();
      depth = frames.get_depth_frame();

      // Generate the pointcloud and texture mappings
      rs_points = pc.calculate(depth);
      auto points = points_to_pcl(rs_points);

      dist_to_center = depth.get_distance(depth.get_width() / 2, depth.get_height() / 2);
      std::cout << "distance to center: " << dist_to_center << std::endl;

      auto center = points->at(points->width / 2, points->height / 2);
      std::cout << "center point is (" << double(center.x) << ", " << double(center.y) << ", " << double(center.z) << ")" << std::endl;
      //      std::cout << "size is " << center[123] << std::endl;
      //[rs_points.size() / 2];
      //std::cout << "center point is (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl;


#if 0
      pcl::IndicesPtr indices (new std::vector <int>);
      pcl::MinCutSegmentation<pcl::PointXYZ> seg;
      seg.setInputCloud (points);
      seg.setIndices (indices);
      pcl::PointCloud<pcl::PointXYZ>::Ptr foreground_points(new pcl::PointCloud<pcl::PointXYZ> ());
      pcl::PointXYZ point;
      point.x = 0.0;
      point.y = 0.0;
      point.z = 0.0;
      foreground_points->points.push_back(point);
      seg.setForegroundPoints (foreground_points);

      seg.setSigma (0.25);
      seg.setRadius (3.0433856);
      seg.setNumberOfNeighbours (14);
      seg.setSourceWeight (0.8);

      std::vector <pcl::PointIndices> clusters;
      seg.extract (clusters);

      std::cout << "Maximum flow is " << seg.getMaxFlow () << std::endl;

      pcl::PointCloud <pcl::PointXYZRGB>::Ptr planar_points = seg.getColoredCloud ();
#endif


#if 0
      std::vector<int> inliers;
      pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr planar (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (points));
      pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(planar);
      ransac.setDistanceThreshold(1);
      ransac.computeModel();
      ransac.getInliers(inliers);

      pcl::PointCloud<pcl::PointXYZ>::Ptr planar_points (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::copyPointCloud(*points, inliers, *planar_points);
#endif

      /*      pcl_ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PassThrough<pcl::PointXYZ> pass;
      pass.setInputCloud(pcl_points);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(0.0, 1.0);
      pass.filter(*cloud_filtered);
      */
      std::vector<pcl_ptr> layers;
      layers.push_back(points);
      draw_pointcloud(app, app_state, layers);

#if 0
      std::vector<pcl_rgbptr> layers2;
      layers2.push_back(planar_points);
      draw_pointcloud(app, app_state, layers2);
#endif

    }

    return EXIT_SUCCESS;
  }
catch (const rs2::error & e)
  {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
catch (const std::exception & e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }


// Registers the state variable and callbacks to allow mouse control of the pointcloud
void register_glfw_callbacks(window& app, state& app_state)
{
  app.on_left_mouse = [&](bool pressed)
  {
    app_state.ml = pressed;
  };

  app.on_mouse_scroll = [&](double xoffset, double yoffset)
  {
    app_state.offset_x += static_cast<float>(xoffset);
    app_state.offset_y += static_cast<float>(yoffset);
  };

  app.on_mouse_move = [&](double x, double y)
  {
    if (app_state.ml)
      {
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

  app.on_key_release = [&](int key)
  {
    if (key == GLFW_KEY_ESCAPE || key == 'Q') {
      glfwSetWindowShouldClose(app, GLFW_TRUE);
    } else if (key == 'R' /* reset */) {
      app_state.yaw = app_state.pitch = 0; app_state.offset_x = app_state.offset_y = 0.0;
    } else {
      printf("key pressed == [%d]\n", key);
    }
  };
}


// Handles all the OpenGL calls needed to display the point cloud
void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points)
{
  // OpenGL commands that prep screen for the pointcloud
  glPopMatrix();
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  float width = app.width(), height = app.height();

  glClearColor(192. / 255, 192. / 255, 192. / 255, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

  glPointSize(width / 640);
  glEnable(GL_TEXTURE_2D);

  int color = 0;

  for (auto&& pc : points)
    {
      auto c = colors[(color++) % (sizeof(colors) / sizeof(float3))];

      glBegin(GL_POINTS);
      glColor3f(c.x, c.y, c.z);

      /* this segment actually prints the pointcloud */
      for (int i = 0; i < pc->points.size(); i++)
        {
          auto&& p = pc->points[i];
          if (p.z)
            {
              // upload the point and texture coordinates only for points we have depth data for
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


void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_rgbptr>& points)
{
  // OpenGL commands that prep screen for the pointcloud
  glPopMatrix();
  glPushAttrib(GL_ALL_ATTRIB_BITS);

  float width = app.width(), height = app.height();

  glClearColor(192. / 255, 192. / 255, 192. / 255, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

  glPointSize(width / 640);
  glEnable(GL_TEXTURE_2D);

  int color = 0;

  for (auto&& pc : points)
    {
      glBegin(GL_POINTS);
      glColor3f(1.0, 0.0, 0.0);

      /* this segment actually prints the pointcloud */
      for (int i = 0; i < pc->points.size(); i++)
        {
          auto&& p = pc->points[i];
          if (p.z)
            {
              // upload the point and texture coordinates only for points we have depth data for
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
