#include <cstddef>
#include <ctime>
#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>


int calculateOptFlow(cv::Mat prev_frame, cv::Mat frame, cv::Mat& output) {
  cv::Mat flow;

    //the algorithm uses gray images
  //  cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
  //  cv::cvtColor(prev_frame,prev_frame,cv::COLOR_BGR2GRAY);

  cv::calcOpticalFlowFarneback(prev_frame, frame, flow,0.4, 1, 3/*12*/, 2, 3/* 8*/, 1.2, 0);

  cv::Mat angle(flow.rows,flow.cols, CV_32FC1);
  cv::Mat dst(flow.rows,flow.cols, CV_32FC1);
  std::vector<cv::Point2f> samples(flow.rows*flow.cols);

  int n=0;
  for(int y=0;y<flow.rows;y++) {
    for(int x=0;x<flow.cols ; x++) {

      angle.at<float>(y,x) = (atan2(flow.at<cv::Point2f>(y,x).y, flow.at<cv::Point2f>(y,x).x));
      samples[n++] =  flow.at<cv::Point2f>(y, x);
    }
  }

  // split into 2 clusters : background and foreground
  cv::Mat labels,centers;
  cv::kmeans(samples,2,labels,cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,10,1.0),3,cv::KMEANS_PP_CENTERS,centers);

  // create a B&W matrix from the labels
  for(int i=0;i<(int)samples.size();i++)
    {
      int row = static_cast<int>(i/dst.cols);
      int col = i%dst.cols;
      if(labels.at<int>(i) == 1) {
        dst.at<float>(row,col) = 255;
      } else {
        dst.at<float>(row,col) = 0;
      }
    }

  //conversion for the use of findContours afterwards
  dst.convertTo(output,CV_8UC1);

  return 1;
}



int main(int argc, char** argv) {

  cv::VideoCapture cap("/Users/morrison/Desktop/sky.mp4");
  int width=640;
  int height=480;
  cv::Mat first_frame;
  cap.read(first_frame);
  cv::resize(first_frame, first_frame, cv::Size(width, height), cv::INTER_LINEAR);

  cv::VideoCapture pew("/Users/morrison/Desktop/asteroids.mp4");

  cv::Mat movement;
  cv::Ptr<cv::BackgroundSubtractor> BS = cv::createBackgroundSubtractorKNN(100, 50, false);

  cv::Mat prev_gray;
  cv::cvtColor(first_frame, prev_gray, cv::COLOR_BGR2RGB);
  //  cv::cvtColor(first_frame, prev_gray, cv::COLOR_BGR2GRAY);

  cv::Mat frame, frame2;
  cap.read(frame);

  int i = 0;
  while(cap.isOpened()) {

    if (i++ % 3 == 0 && !cap.read(frame))
      exit(0);
    if (!pew.read(frame2))
      exit(1);

    cv::resize(frame, frame, cv::Size(width, height), cv::INTER_LINEAR);
    cv::resize(frame2, frame2, cv::Size(width, height), cv::INTER_LINEAR);

    cv::imshow("input", frame);

    //    cv::Mat gray;
    //    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    BS->apply(frame, movement);
    cv::erode(movement, movement, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1,1)));
    cv::cvtColor(movement, movement, cv::COLOR_GRAY2RGB);
    //    cv::dilate(movement, movement, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    //    cv::multiply(movement, 100, movement);
    //    cv::inRange(movement, cv::Scalar(64), cv::Scalar(255), movement);
    // cv::multiply(movement, .10, movement);

    cv::GaussianBlur(movement, movement, cv::Size(5,5), 3, 3);
    cv::GaussianBlur(movement, movement, cv::Size(5,5), 3, 3);
    //    medianBlur(movement, movement, 3);
    //    cv::floodFill(movement, cv::Point(0,0), 255);
    //    cv::threshold(movement, movement, 200, 255, cv::THRESH_BINARY);
    //    cv::threshold(movement, movement, 50, 0, cv::THRESH_BINARY);
    //    cv::Mat masked;
    //    cv::bitwise_or(movement, movement, frame2);
    //cv::bitwise_not(movement, movement);
    //    movement = (frame | frame2);

#if 0
  cv::Mat bgr;
    cv::absdiff(prev_gray, gray, bgr);
    cv::multiply(bgr, 10, bgr);
    cv::add(bgr, 100, bgr);
    /*    cv::GaussianBlur(bgr, bgr, cv::Size(5,5), 3, 3);
    cv::GaussianBlur(bgr, bgr, cv::Size(5,5), 3, 3);
    cv::GaussianBlur(bgr, bgr, cv::Size(5,5), 3, 3);
    */
    cv::inRange(bgr, cv::Scalar(100), cv::Scalar(255), bgr);
    cv::subtract(bgr, 100, bgr);
    cv::multiply(bgr, .10, bgr);

    /*    medianBlur(bgr, bgr, 3);
    medianBlur(bgr, bgr, 3);
    medianBlur(bgr, bgr, 3);
    */

    //    cv::inRange(bgr, cv::Scalar(200), cv::Scalar(255), bgr);
    //    cv::threshold(bgr, bgr, 100, 0, cv::THRESH_BINARY);
    //    cv::floodFill(bgr, cv::Point(0,0), 255);

    //        cv::threshold(bgr, bgr, 3, 128, cv::THRESH_BINARY);
    //    cv::erode(bgr, bgr, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));
    //    cv::dilate(bgr, bgr, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

    calculateOptFlow(prev_gray, gray, bgr);

    cv::Mat flow(prev_gray.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(prev_gray, gray,
                                 flow,
                                 0.5, 3, 5 /*15*/, 3, 5, 1.2, 0);

    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0, 1.0, cv::NORM_MINMAX);
    angle *= 1. / 360. * 180. / 255.;

    cv::Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
    cv::cvtColor(bgr, bgr, cv::COLOR_BGR2GRAY);
    cv::threshold(bgr, bgr, 25, 255, cv::THRESH_BINARY);
#endif

    std::stringstream ss;
    rectangle(movement, cv::Point(10, 2), cv::Point(100,20), cv::Scalar(255,255,255), -1);
    ss << cap.get(cv::CAP_PROP_POS_FRAMES);
    std::string frameNumberString = ss.str();
    cv::putText(movement, frameNumberString.c_str(), cv::Point(15, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));

    cv::imshow("output", movement);

    prev_gray = gray;

    int key = cv::waitKey(1);
    if (key == 'q' || key == 27)
      break;
  }

  cap.release();
  cv::destroyAllWindows();

  return 0;
}
