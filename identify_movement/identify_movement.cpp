#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <librealsense2/rs.hpp>


static float get_depth_scale(rs2::device dev)
{
  // Go over the device's sensors
  for (rs2::sensor& sensor : dev.query_sensors())
  {
    // Check if the sensor if a depth sensor
    if (rs2::depth_sensor dpt = rs2::depth_sensor(sensor))
    {
      return dpt.get_depth_scale();
    }
  }
  throw std::runtime_error("Device does not have a depth sensor");
}

static rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
  //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
  //We prioritize color streams to make the view look better.
  //If color is not available, we take another stream that (other than depth)
  rs2_stream align_to = RS2_STREAM_ANY;
  bool depth_stream_found = false;
  bool color_stream_found = false;

  for (rs2::stream_profile sp : streams)
  {
    rs2_stream profile_stream = sp.stream_type();
    if (profile_stream != RS2_STREAM_DEPTH)
    {
      if (!color_stream_found) //Prefer color
      align_to = profile_stream;

  		if (profile_stream == RS2_STREAM_COLOR)
  		{
  			color_stream_found = true;
  		}
    }
  	else
  	{
  		depth_stream_found = true;
  	}
  }

  if (!depth_stream_found)
  	throw std::runtime_error("No Depth stream available");

  if (align_to == RS2_STREAM_ANY)
  	throw std::runtime_error("No stream found to align with Depth");

  return align_to;
}


bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
  for(auto&& sp : prev)
  {
    //If previous profile is in current (maybe just added another)
    auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
    if (itr == std::end(current)) //If it previous stream wasn't found in current
    {
      return true;
    }
  }
  return false;
}


static int rs_read(rs2::pipeline_profile profile, rs2::pipeline p, cv::Mat& img, cv::Mat &dimg) {
    rs2::frameset frames = p.wait_for_frames();

    rs2_stream align_to = find_stream_to_align(profile.get_streams());
    rs2::align align(align_to);
    auto processed = align.process(frames);

    rs2::frame color = frames.get_color_frame();
    const int w = color.as<rs2::video_frame>().get_width();
    const int h = color.as<rs2::video_frame>().get_height();
    cv::Mat nimg(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
    cv::cvtColor(nimg, img, cv::COLOR_BGR2RGB);

    rs2::frame depth = frames.get_depth_frame();
    const int dw = depth.as<rs2::video_frame>().get_width();
    const int dh = depth.as<rs2::video_frame>().get_height();

    cv::Mat ndimg(cv::Size(dw, dh), CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
    cv::resize(ndimg, dimg, cv::Size(w, h), cv::INTER_LINEAR);
    ndimg.convertTo(ndimg, CV_8UC1, 15 / 256.0);
    dimg = ndimg;

    return 1;
}


static bool
cmpContourAreas(std::vector<cv::Point> c1, std::vector<cv::Point> c2 ) {
    double i = fabs(contourArea(cv::Mat(c1)));
    double j = fabs(contourArea(cv::Mat(c2)));
    return (i > j);
}


int main(int ac, char *av[]) {
  int k = 0;

  const char *sourceRGB = "Source";
  const char *outputRGB = "Output";
  const char *maskKNN = "KNN Mask";
  const char *maskMOG = "MOG2 Mask";
  const char *outputKNN = "KNN Output";
  const char *outputMOG = "MOG Output";
  const char *sourceDST = "Depth";

  int history = 100;
  double thresh = 400;
  bool shadows = true;
  cv::Ptr<cv::BackgroundSubtractor> KNN = cv::createBackgroundSubtractorKNN(history, thresh, shadows);
  cv::Ptr<cv::BackgroundSubtractor> MOG = cv::createBackgroundSubtractorMOG2(history, sqrt(thresh)/4.0, shadows);

  cv::Mat initial;
  cv::Mat dsrc = cv::Mat::zeros(initial.size(), CV_8UC1);
  rs2::pipeline pipe;

  rs2::config cfg;
  cfg.enable_stream(rs2_stream::RS2_STREAM_DEPTH, 1280, 720, rs2_format::RS2_FORMAT_Z16);
  //change the color format to BGR8 for opencv
  cfg.enable_stream(rs2_stream::RS2_STREAM_COLOR, 1280, 720, rs2_format::RS2_FORMAT_RGB8);

  
  rs2::pipeline_profile profile = pipe.start(cfg);

  /* get units for depth pixels */
  float depth_scale = get_depth_scale(profile.get_device());
  rs2_stream align_to = find_stream_to_align(profile.get_streams());
  rs2::align align(align_to);
  float depth_clipping_distance = 10000.f;

  /* actually read our realsense data */
  rs_read(profile, pipe, initial, dsrc);

  cv::Mat src = cv::Mat::zeros(initial.size(), CV_8UC3);
  cv::Mat out = cv::Mat::zeros(initial.size(), CV_8UC3);
  cv::Mat knnMask = cv::Mat::zeros(initial.size(), CV_8UC1);
  cv::Mat mogMask = cv::Mat::zeros(initial.size(), CV_8UC1);
  cv::Mat knnOut = cv::Mat::zeros(initial.size(), CV_8UC3);
  cv::Mat mogOut = cv::Mat::zeros(initial.size(), CV_8UC3);

  initial = cv::Mat::zeros(initial.size(), CV_8UC3);

  /* initialize window positions */
  cv::imshow(sourceRGB, src);
  cv::moveWindow(outputRGB, 0, 0);
  cv::imshow(outputRGB, out);
  cv::moveWindow(outputRGB, 0, 505);
  cv::imshow(maskKNN, knnMask);
  cv::moveWindow(maskKNN, 640, 0);
  cv::imshow(maskMOG, mogMask);
  cv::moveWindow(maskMOG, 1280, 0);
  cv::imshow(outputKNN, knnOut);
  cv::moveWindow(outputKNN, 640, 505);
  cv::imshow(outputMOG, mogOut);
  cv::moveWindow(outputMOG, 1280, 505);
  cv::imshow(sourceDST, dsrc);
  cv::moveWindow(sourceDST, 0, 1010);

  while (k != 'q' && k != 27) {
    fflush(stdout);
    rs_read(profile, pipe, src, dsrc);
    // std::cout << "Width : " << src.size().width << std::endl;
    // std::cout << "Height: " << src.size().height << std::endl;

    /* check if realsense config changed */
    if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams())) {
      // if profile changed, update align object and get new depth scale
      profile = pipe.get_active_profile();
      align_to = find_stream_to_align(profile.get_streams());
      align = rs2::align(align_to);
      depth_scale = get_depth_scale(profile.get_device());
    }

  /* do fun stuff */
    KNN->apply(src, knnMask);
    MOG->apply(src, mogMask);

    /* apply some thresholding */
    cv::threshold(knnMask, knnMask, 200, 255, cv::THRESH_BINARY);
    cv::threshold(mogMask, mogMask, 200, 255, cv::THRESH_BINARY);

    /* reduce noise */
    cv::erode(knnMask, knnMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::dilate(knnMask, knnMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::erode(mogMask, mogMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::dilate(mogMask, mogMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

    std::vector<std::vector<cv::Point> > knnContours;
    std::vector<cv::Vec4i> knnHierarchy;
    cv::findContours(knnMask, knnContours, knnHierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // std::cout << "contours found: " << knnContours.size() << std::endl;

    /* reset output buffers */
    src.copyTo(out);
    initial.copyTo(knnOut);
    initial.copyTo(mogOut);

    /* sort by area */
    std::sort(knnContours.begin(), knnContours.end(), cmpContourAreas);

     /* Don't consider anything less than 10px^2 or bigger than a 1/3
      * of the view.
      */
    double minArea = 100.0;
    double maxArea = src.size().width * src.size().height / 3.0;
    std::vector<cv::Rect> knnBoundRect( knnContours.size() );
    std::vector<std::vector<cv::Point> > knnContoursPoly( knnContours.size() );

    for (size_t i = 0; i < knnContours.size(); i++) {
      if (cv::contourArea(knnContours[i]) > minArea && cv::contourArea(knnContours[i]) < maxArea) {
        // std::cout << "displaying contour " << i << " with size " << cv::contourArea(knnContours[i]) << std::endl;
        cv::approxPolyDP(knnContours[i], knnContoursPoly[i], 3, true);
        knnBoundRect[i] = cv::boundingRect(knnContours[i]);
        cv::drawContours(out, knnContoursPoly, (int)i, cv::Scalar(255,0,0), 2);
        cv::rectangle(out, knnBoundRect[i].tl(), knnBoundRect[i].br(), cv::Scalar(0,0,255), 2);
        if (i > 3)
          break;
      }
    }

    /* get the masked rgb */
    cv::bitwise_and(src, src, knnOut, knnMask);
    cv::bitwise_and(src, src, mogOut, mogMask);

    //    std::vector<std::vector<cv::Point> > mogContoursPoly( mogContours.size() );
    //    std::vector<Rect> mogBoundRect( mogContours.size() );

#if 0
    /* convert mask back to color */
    cv::cvtColor(knnMask, knnMask, cv::COLOR_GRAY2BGR);
    cv::cvtColor(mogMask, knnMask, cv::COLOR_GRAY2BGR);
#endif

    /* display */
    cv::imshow(sourceRGB, src);
    cv::imshow(outputRGB, out);
    cv::imshow(maskKNN, knnMask);
    cv::imshow(maskMOG, mogMask);
    cv::imshow(outputKNN, knnOut);
    cv::imshow(outputMOG, mogOut);
    cv::imshow(sourceDST, dsrc);

    k = cv::waitKey(200);
  }

  cv::destroyWindow(sourceRGB);

  return 0;
}
