#ifndef __IMG_PROCESS_H__
#define __IMG_PROCESS_H__

// #include "frame.h"
#include "point_matcher.h"
#include "feature_detector.h"
#include "common_lib.h"
// #include "map.h"
#include <vikit/abstract_camera.h>

#include <pcl/kdtree/kdtree_flann.h>

typedef pcl::PointCloud<PointType> PointCloud;

typedef pcl::PointCloud<PointType>::Ptr PointCloudPtr;

struct CamObs{
  std::vector<PointType> points;
  Eigen::Matrix<float, 259, Eigen::Dynamic> features;
  std::vector<bool> is_valid;
  StatesGroup state;
  cv::Mat image;
};

struct InputData{
  double time;
  cv::Mat image;
  PointCloudPtr pointcloud;

  InputData() {}
  InputData& operator =(InputData& other){
		time = other.time;
		image = other.image.clone();
		return *this;
	}
};

class ImgProcess{

public:
  ImgProcess();
  ImgProcess(const ImgConfigs& configs, vk::AbstractCamera* cam);
  void addInputData(const InputData& data );
//   void matches();

  bool getDownSamplePointCloud(PointCloud &pointcloud);
  void getPointByDSPointCloud();
  bool checkPointPlane(PointType &point);
  double getHuberLossScale( double reprojection_error, double outlier_threshold = 1.0 );
  Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d &vector); 

  void updateState();
  void outlierRejectF(const Eigen::Matrix<float, 259, Eigen::Dynamic>& ref_features,
                      const Eigen::Matrix<float, 259, Eigen::Dynamic>& cur_features,
                      const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& matches_good);

  void distributePointsByDepth(const std::vector<PointType>& _points, std::vector<bool>& _is_valid, size_t target_count);

  V3F getpixel(cv::Mat img, V2D pc);
  pcl::KdTreeFLANN<PointType>::Ptr _kdtree;
  PointCloudPtr unit_sphere_pointcloud_;

//   Eigen::Matrix<float, 259, Eigen::Dynamic> _candidate_features;
  std::vector<Eigen::Vector2d> _last_keypoints;
  Eigen::Matrix<float, 259, Eigen::Dynamic> _features;
  std::vector<Eigen::Vector2d> _keypoints;

  std::vector<PointType> _points;
  std::vector<bool> _is_valid;
  std::deque<CamObs> _cam_obs;
  StatesGroup* state;
  StatesGroup* state_propagat;
  MatrixXd H_sub;
  Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  VectorXd z;
  M3D R_c2i;
  V3D T_c2i;
  // vk::robust_cost::WeightFunctionPtr weight_function_;
  cv::Mat _last_keyimage;
  PointMatcherPtr _point_matcher;
  FeatureDetectorPtr _feature_detector;
  std::deque<cv::Mat> _keyimages;
  vk::AbstractCamera*    _cam;                   //!< Camera model.
  std::deque<PointCloud> _clouds;
  std::deque<StatesGroup> _clouds_state;
  int frame_winsize;
  double weight_scale_unit;
  double fx,fy,cx,cy;
  double reprojection_cov;
  size_t CLOUD_SIZE_USED;
  int max_iter;
  cv::Mat img_draw,image_now;
  double horizontal_fov,vertical_fov;
  double min_angular;
};
typedef std::shared_ptr<ImgProcess> ImgProcessPtr;



#endif