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
  const char *maskKNN = "KNN Mask";
  const char *maskMOG = "MOG2 Mask";
  const char *outputKNN = "KNN Output";
  const char *outputMOG = "MOG Output";

  int history = 5;
  double thresh = 400;
  bool shadows = false;
  cv::Ptr<cv::BackgroundSubtractor> KNN = cv::createBackgroundSubtractorKNN(history, thresh, shadows);
  cv::Ptr<cv::BackgroundSubtractor> MOG = cv::createBackgroundSubtractorMOG2(history, sqrt(thresh), shadows);

  rs2::pipeline pipe;
  pipe.start();

  cv::Mat src;
  cv::Mat knnMask;
  cv::Mat mogMask;
  cv::Mat prev;

  while (k != 'q' && k != 27) {
    fflush(stdout);
    rs_read(pipe, src);
    // std::cout << "Width : " << src.size().width << std::endl;
    // std::cout << "Height: " << src.size().height << std::endl;

    /* do fun stuff */
    KNN->apply(src, knnMask);
    MOG->apply(src, mogMask);

    /* reduce noise */
    cv::erode(knnMask, knnMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::dilate(knnMask, knnMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::erode(mogMask, mogMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::dilate(mogMask, mogMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

    cv::Mat knnOut, mogOut;
    cv::bitwise_and(src, src, knnOut, knnMask);
    cv::bitwise_and(src, src, mogOut, mogMask);

    /* display */
    cv::imshow(sourceRGB, src);
    cv::imshow(maskKNN, knnMask);
    cv::imshow(maskMOG, mogMask);
    cv::imshow(outputKNN, knnOut);
    cv::imshow(outputMOG, mogOut);
    cv::moveWindow(maskKNN, 640, 0);
    cv::moveWindow(maskMOG, 1280, 0);
    cv::moveWindow(outputKNN, 640, 505);
    cv::moveWindow(outputMOG, 1280, 505);

    k = cv::waitKey(200);
    prev = src;
  }

  cv::destroyWindow(sourceRGB);

  return 0;
}
