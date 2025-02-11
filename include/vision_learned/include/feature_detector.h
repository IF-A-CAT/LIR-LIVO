#ifndef FEATURE_DETECTOR_H_
#define FEATURE_DETECTOR_H_

#include "super_point.h"
#include "plnet.h"


class FeatureDetector{
public:
  FeatureDetector(const PLNetConfig& plnet_config);

  bool removewithSemantic(const cv::Mat& image_semantic, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, std::vector<Eigen::Vector4d>& lines);

  bool Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features);
  bool Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, std::vector<Eigen::Vector4d>& lines);
  bool Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions);

  bool Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, const cv::Mat& semantic_img);


  bool Detect(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
      Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features);

  bool Detect(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
      Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features, std::vector<Eigen::Vector4d>& left_lines, 
      std::vector<Eigen::Vector4d>& right_lines);

  bool Detect(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
      Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features, std::vector<Eigen::Vector4d>& left_lines, 
      std::vector<Eigen::Vector4d>& right_lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions);

  bool Detect(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
      Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features, std::vector<Eigen::Vector4d>& left_lines, 
      std::vector<Eigen::Vector4d>& right_lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, const cv::Mat& sematic_img);
private:
  PLNetConfig _plnet_config;
  SuperPointPtr _superpoint;
  PLNetPtr _plnet;
};

typedef std::shared_ptr<FeatureDetector> FeatureDetectorPtr;

#endif  // FEATURE_DETECTOR_H_ 
