/* Copyright - Benjamin Laugraud <blaugraud@ulg.ac.be> - 2016
 * Copyright - Marc Van Droogenbroeck <m.vandroogenbroeck@ulg.ac.be> - 2016
 *
 * ViBe is covered by a patent (see http://www.telecom.ulg.ac.be/research/vibe).
 *
 * Permission to use ViBe without payment of fee is granted for nonprofit
 * educational and research purposes only.
 *
 * This work may not be copied or reproduced in whole or in part for any
 * purpose.
 *
 * Copying, reproduction, or republishing for any purpose shall require a
 * license. Please contact the authors in such cases. All the code is provided
 * without any guarantee.
 *
 * This simple example program takes a path to a video sequence as an argument.
 * When it is executed, two windows are opened: one displaying the input
 * sequence, and one displaying the segmentation maps produced by ViBe. Note
 * that this program uses the a polychromatic version of ViBe with 3 channels.
 */
#include <cstddef>
#include <ctime>
#include <iostream>

#ifndef    OPENCV_3
#include <cv.h>
#include <highgui.h>
#else
#include <opencv2/opencv.hpp>
#endif  /* OPENCV_3 */

#include <libvibe++/ViBe.h>
#include <libvibe++/distances/Manhattan.h>
#include <libvibe++/system/types.h>

using namespace std;
using namespace cv;
using namespace ViBe;


int main(int argc, char** argv) {
  if (argc != 2) {
    cerr << "A video file must be given as an argument to the program!";
    cerr << endl;

    return EXIT_FAILURE;
  }

  /* Parameterization of ViBe. */
  typedef ViBeSequential<3, Manhattan<3> >                                ViBe;

  /* Random seed. */
  srand(time(NULL));

  cv::VideoCapture decoder(argv[1]);
  cv::VideoCapture sky;
  sky.open("/Users/morrison/Desktop/sky.mp4");

  cv::Mat frame;

  int32_t height = decoder.get(CAP_PROP_FRAME_HEIGHT);
  int32_t width  = decoder.get(CAP_PROP_FRAME_WIDTH);

  ViBe* vibe = NULL;
  cv::Mat segmentationMap(height, width, CV_8UC1);
  bool firstFrame = true;

  int cnt = 0;
  cv::Mat bg;
  sky.read(bg);

  while (decoder.read(frame)) {
    if (firstFrame) {
      /* Instantiation of ViBe. */
      vibe = new ViBe(height, width, frame.data);
      firstFrame = false;
    }

    /* Segmentation and update. */
    vibe->segmentation(frame.data, segmentationMap.data);
    vibe->update(frame.data, segmentationMap.data);

    /* Post-processing: 3x3 median filter. */
    medianBlur(segmentationMap, segmentationMap, 3);

    int erosion_elem = 0;
    int erosion_type = 0;
    int erosion_size = 2;
    if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement( erosion_type,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );
    Mat dilated;
    cv::dilate( segmentationMap, dilated, element );
    cv::imshow( "Dilation Demo", dilated );
    dilated.copyTo(segmentationMap);

    /* slow down timelapse background video */
    if (cnt++ % 4 == 0)
      sky.read(bg);

    cv::Mat edgesNeg = segmentationMap;
    cv::floodFill(edgesNeg, cv::Point(0,0), 255);
    bitwise_not(edgesNeg, edgesNeg);
    segmentationMap = (edgesNeg | segmentationMap);
    Mat nimg;
    cv::cvtColor(segmentationMap, nimg, cv::COLOR_GRAY2RGB);
    //    cv::cvtColor(bg, bg, cv::COLOR_BGR2RGB);
    const int w = segmentationMap.cols;
    const int h = segmentationMap.rows;
    // std::cout << "o Width : " << nimg.size().width << std::endl;
    // std::cout << "o Height: " << nimg.size().height << std::endl;
    // std::cout << "bg Width : " << bg.size().width << std::endl;
    // std::cout << "bg Height: " << bg.size().height << std::endl;

    cv::Mat cropped;
    cropped = bg(cv::Rect(0, 0, w*4.0, h*4.0));
    cv::resize(cropped, cropped, cv::Size(w, h), INTER_LINEAR);
    // cv::rotate(bg, bg, 90);
    nimg = (nimg | cropped);

    imshow("Input video", frame);
    imshow("Segmentation by ViBe", nimg /*segmentationMap */);

    waitKey(1);
  }

  delete vibe;

  destroyAllWindows();
  decoder.release();

  return EXIT_SUCCESS;
}
