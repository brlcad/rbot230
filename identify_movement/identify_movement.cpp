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

  int history = 100;
  double thresh = 400;
  bool shadows = true;
  cv::Ptr<cv::BackgroundSubtractor> KNN = cv::createBackgroundSubtractorKNN(history, thresh, shadows);
  cv::Ptr<cv::BackgroundSubtractor> MOG = cv::createBackgroundSubtractorMOG2(history, sqrt(thresh)/4.0, shadows);

  rs2::pipeline pipe;
  pipe.start();

  cv::Mat initial;
  rs_read(pipe, initial);

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

  while (k != 'q' && k != 27) {
    fflush(stdout);
    rs_read(pipe, src);
    // std::cout << "Width : " << src.size().width << std::endl;
    // std::cout << "Height: " << src.size().height << std::endl;

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

    k = cv::waitKey(200);
  }

  cv::destroyWindow(sourceRGB);

  return 0;
}
