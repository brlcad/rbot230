#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include <librealsense2/rs.hpp>


static int rs_read(rs2::pipeline p, cv::Mat& img) {
    rs2::frameset frames = p.wait_for_frames();
    rs2::frame color = frames.get_color_frame();
    const int w = color.as<rs2::video_frame>().get_width();
    const int h = color.as<rs2::video_frame>().get_height();
    cv::Mat nimg(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
    cv::cvtColor(nimg, img, cv::COLOR_BGR2RGB);
    return 1;
}


int main(int ac, char *av[]) {
  int k = 0;

  const char *sourceRGB = "Source";
  const char *maskGRAY = "Mask";
  const char *outputRGB = "Output";

  cv::Ptr<cv::BackgroundSubtractor> BS = cv::createBackgroundSubtractorKNN(100,
 50, false);

  rs2::pipeline pipe;
  pipe.start();

  cv::Mat src;
  cv::Mat mask;
  cv::Mat prev;

  while (k != 'q' && k != 27) {
    fflush(stdout);
    rs_read(pipe, src);
    // std::cout << "Width : " << src.size().width << std::endl;
    // std::cout << "Height: " << src.size().height << std::endl;

    /* do fun stuff */
    BS->apply(src, mask);

    cv::imshow(sourceRGB, src);
    cv::imshow(maskGRAY, mask);
    cv::moveWindow(maskGRAY, 640, 0);

    k = cv::waitKey(200);
    prev = src;
  }

  cv::destroyWindow(sourceRGB);

  return 0;
}
