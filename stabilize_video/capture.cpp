#include <iostream>
#include <algorithm>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <librealsense2/rs.hpp>



int main(int ac, char *av[]) {
  int k = 0;
  cv::Mat prevImg;

  const char *windowRGB = "Image Viewer";
  const char *windowEQ = "Equalized Viewer";
  const char *windowMAT = "Matching Points";
  const char *windowWAR = "Warped";

  rs2::pipeline pipe;
  pipe.start();

  cv::namedWindow(windowRGB);
  cv::moveWindow(windowRGB, 0, 0);
  cv::namedWindow(windowEQ);
  cv::moveWindow(windowEQ, 640, 0);
  cv::namedWindow(windowWAR);
  cv::moveWindow(windowWAR, 1280, 0);
  cv::namedWindow(windowMAT);
  cv::moveWindow(windowMAT, 0, 512);

  cv::Ptr<cv::Feature2D> orb = cv::ORB::create(20);
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  while (k != 'q' && k != 27) {
    fflush(stdout);
    rs2::frameset frames = pipe.wait_for_frames();
    rs2::frame color = frames.get_color_frame();

    const int w = color.as<rs2::video_frame>().get_width();
    const int h = color.as<rs2::video_frame>().get_height();

    /* show our color rgb image */
    cv::Mat img(cv::Size(w, h), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::imshow(windowRGB, img);

    /* show our equalized b&w image */
    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
    cv::equalizeHist(img, img);
    cv::imshow(windowEQ, img);

    if (!prevImg.cols)
      prevImg = img;

    /* match features of prev b&w to current */
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(img, cv::Mat(), keypoints1, desc1);
    orb->detectAndCompute(prevImg, cv::Mat(), keypoints2, desc2);

    // std::cout << "types are " << img.cols << " and " << prevImg.cols << std::endl;

    std::vector<cv::DMatch> matches;
    matcher->match(desc1, desc2, matches, cv::Mat());
    std::sort(matches.begin(), matches.end());

    /* filter out the bottom-half */
    double thresh = 0.5;
    matches.erase(matches.begin()+(matches.size()*thresh), matches.end());

    cv::Mat img_matches;
    cv::drawMatches(img, keypoints1, prevImg, keypoints2, matches, img_matches);
    cv::imshow(windowMAT, img_matches);

    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
      points1.push_back(keypoints1[matches[i].queryIdx].pt);
      points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    if (matches.size() >= 5) {
      cv::Mat imgReg;
      cv::Mat homo = cv::findHomography(points1, points2, cv::RANSAC);
      cv::warpPerspective(img, imgReg, homo, img.size());
      cv::imshow(windowWAR, imgReg);
    }

    k = cv::waitKey(200);

    prevImg = img;
  }

  cv::destroyWindow(windowRGB);
  cv::destroyWindow(windowEQ);
  return 0;
}

