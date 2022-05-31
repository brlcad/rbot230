#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>


int main(int ac, char *av[]) {

  if (ac != 2) {
    std::cout << "Usage: " << av[0] << " {video|'rs'}" << std::endl;
    return 1;
  }

#if 1
  auto model = cv::dnn::readNet("frozen_inference_graph.pb",
                                "ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
                                "Tensorflow");
  std::ifstream ifs("object_detection_classes_coco.txt");
#else
  auto model = cv::dnn::readNet("mobilenet_v2_deploy.prototxt",
                                "mobilenet_v2.caffemodel",
                                "Caffe");
  std::ifstream ifs("synset.txt");
#endif

  std::vector<std::string> class_names;
  std::string line;
  while (getline(ifs, line)) {
    class_names.push_back(line);
  }

  //  cv::Mat image = cv::imread("minivan.jpg");
  cv::VideoCapture cap("videos/city/city.mp4");//test_video.mp4");
  while (cap.isOpened()) {

    cv::Mat image;
    bool success = cap.read(image);
    if (!success)
      break;

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), cv::Scalar(103.94, 116.78, 123.68), true, false);//, 0.017);//, cv::Size(224, 224));//, cv::Scalar(103.94, 116.78, 123.68));

    cv::Point position;
    double probability;

    model.setInput(blob);
    cv::Mat outputs = model.forward();

    //    std::cout << "outputs: [2]=" << outputs.size[2] << " and [3]=" << outputs.size[3] << std::endl;

    cv::Mat detectionMat(outputs.size[2], outputs.size[3], CV_32F, outputs.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
      int id = detectionMat.at<float>(i, 1);
      float conf = detectionMat.at<float>(i, 2);
      //      std::cout << "conf=" << conf << std::endl;

      if (class_names[id-1] != "car")
        continue;

      // Check if the detection is of good quality
      if (conf > 0.3) {
        int box_x = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
        int box_y = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
        int box_width = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols - box_x);
        int box_height = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows - box_y);
        cv::rectangle(image, cv::Point(box_x, box_y), cv::Point(box_x+box_width, box_y+box_height), cv::Scalar(255, 255, 255), 2);

        cv::putText(image, class_names[id-1].c_str(), cv::Point(box_x, box_y-5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
      }

    }

#if 0
    cv::minMaxLoc(outputs.reshape(1, 1), 0, &probability, 0, &position);
    int idx = position.x;
    std::cout << "position=(" << position.x << ", " << position.y << ")" << std::endl;
    for (int i = 0; i < 5; i++) {
      std::cout << "Predicted #" << i << ": " << probability*100.0 << " => " << class_names[idx].c_str() << std::endl;
    }
    std::string label = cv::format("%s (%3.0f%%)", class_names[idx].c_str(), probability*100.0);
    //  std::cout << "Predicted: " << probability*100.0 << " => " << class_names[idx].c_str() << std::endl;

    cv::putText(image, label, cv::Point(25, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
#endif

    cv::imshow("Detect Cars", image);
    int k = cv::waitKey(10);
    if ((k == 113) || (k == 'q')) {
      break;
    }
  }
  cv::destroyAllWindows();

  return 0;
}
