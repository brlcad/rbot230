#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

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

  if (ac != 2) {
    std::cout << "Usage: " << av[0] << " {video|'rs'}" << std::endl;
    return 1;
  }

#if 1
  auto model = cv::dnn::readNet("nets/mobilenet_v2_tensorflow/frozen_inference_graph.pb",
                                "nets/mobilenet_v2_tensorflow/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
                                "Tensorflow");
  std::ifstream ifs("nets/mobilenet_v2_tensorflow/object_detection_classes_coco.txt");
#else
  auto model = cv::dnn::readNet("nets/mobilenet_v2_caffe/mobilenet_v2_deploy.prototxt",
                                "nets/mobilenet_v2_caffe/mobilenet_v2.caffemodel",
                                "Caffe");
  std::ifstream ifs("nets/mobilenet_v2_caffe/synset.txt");
#endif

  std::vector<std::string> class_names;
  std::string line;
  while (getline(ifs, line)) {
    class_names.push_back(line);
  }

  typedef enum {
    SRC_RS = 0,
    SRC_IM = 1,
    SRC_VC = 2
  } source_t;

  /* three potential input sources */
  source_t src = SRC_VC;
  rs2::pipeline pipe;
  cv::VideoCapture cap;
  cv::Mat image;
  cv::Mat boxed;

  if (std::string(av[1]) == "rs") {
    std::cout << "Using RealSense sensor as input" << std::endl;
    src = SRC_RS;
    pipe.start();
  } else {
    std::cout << "Using [ " << av[1] << " ] as input" << std::endl;

    image = cv::imread(av[1]);
    if (image.data != NULL) {
      src = SRC_IM;
    } else {
      src = SRC_VC;
      int opened = cap.open(av[1]);
      if (!opened) {
        std::cout << "ERROR: unable to open " << av[1] << " for reading" << std::endl;
      }
    }
  }

  while (1) {
    if (src == SRC_RS)
      rs_read(pipe, image);
    if (src == SRC_VC) {
      if (cap.isOpened()) {
        bool success = cap.read(image);
        if (!success)
          break;
      }
    }

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), cv::Scalar(103.94, 116.78, 123.68), true, false);
    //, 0.017);//, cv::Size(224, 224));//, cv::Scalar(103.94, 116.78, 123.68));

    cv::Point position;

    model.setInput(blob);
    cv::Mat outputs = model.forward();

    //    std::cout << "outputs: [2]=" << outputs.size[2] << " and [3]=" << outputs.size[3] << std::endl;

    cv::Mat detectionMat(outputs.size[2], outputs.size[3], CV_32F, outputs.ptr<float>());
    boxed = image;
    for (int i = 0; i < detectionMat.rows; i++) {
      int id = detectionMat.at<float>(i, 1);
      float conf = detectionMat.at<float>(i, 2);
      //      std::cout << "conf=" << conf << std::endl;

      if (class_names[id-1] != "car")
        continue;

      // Check if the detection is of decent quality
      if (conf > 0.3) {
        int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
        int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
        int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
        int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);
        cv::rectangle(boxed, cv::Point(box_x, box_y), cv::Point(box_x+box_width, box_y+box_height), cv::Scalar(255, 255, 255), 2);

        cv::putText(boxed, class_names[id-1].c_str(), cv::Point(box_x, box_y-5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
      }

    }


#if 0
    double probability;
    cv::minMaxLoc(outputs.reshape(1, 1), 0, &probability, 0, &position);
    int idx = position.x;
    std::cout << "position=(" << position.x << ", " << position.y << ")" << std::endl;
    for (int i = 0; i < 5; i++) {
      std::cout << "Predicted #" << i << ": " << probability*100.0 << " => " << class_names[idx].c_str() << std::endl;
    }
    std::string label = cv::format("%s (%3.0f%%)", class_names[idx].c_str(), probability*100.0);
    //  std::cout << "Predicted: " << probability*100.0 << " => " << class_names[idx].c_str() << std::endl;

    cv::putText(boxed, label, cv::Point(25, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
#endif

    cv::imshow("Detect Cars", boxed);
    int k = cv::waitKey(1000);
    if ((k == 113) || (k == 'q')) {
      std::cout << "k is " << k << std::endl;
      break;
    }
  }

  cv::destroyAllWindows();

  return 0;
}
