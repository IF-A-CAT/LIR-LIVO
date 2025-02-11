#ifndef __CONFIG_STRUCTS_H__
#define __CONFIG_STRUCTS_H__

#include <vector>
#include <string>

struct PointMatcherConfig {
  PointMatcherConfig() {
    matcher = 0;
    image_width = 888;
    image_height = 480;
    onnx_file = "superpoint_lightglue.onnx";
    engine_file = "superpoint_lightglue.engine";
}

  int matcher;
  int image_width;
  int image_height;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct PLNetConfig{
  std::string superpoint_onnx;
  std::string superpoint_engine;

  std::string plnet_s0_onnx;
  std::string plnet_s0_engine;
  std::string plnet_s1_onnx;
  std::string plnet_s1_engine;
  std::string semantic_onnx;
  std::string semantic_engine;

  int use_superpoint;
  int use_semantic;

  int max_keypoints;
  float keypoint_threshold;
  int remove_borders;

  float line_threshold;
  float line_length_threshold;

  PLNetConfig() {
    use_superpoint = 1;
    use_semantic = 0;

    max_keypoints = 450;
    keypoint_threshold = 0.004;
    remove_borders = 4;

    line_threshold = 0.1;
    line_length_threshold: 50;
  }

  void SetModelPath(std::string model_dir){
    if(use_superpoint){
      superpoint_onnx = model_dir + "/superpoint_v1_sim_int32.onnx";
      superpoint_engine = model_dir + "/superpoint_v1_sim_int32.engine";
    }
    
    if(use_semantic){
      semantic_onnx = model_dir + "/semantic.onnx";
      semantic_engine = model_dir + "/semantic.engine";
    }

    plnet_s0_onnx = model_dir + "/plnet_s0.onnx";
    plnet_s0_engine = model_dir + "/plnet_s0.engine";
    plnet_s1_onnx = model_dir + "/plnet_s1.onnx";
    plnet_s1_engine = model_dir + "/plnet_s1.engine";
  }
};


struct SuperPointConfig {
  SuperPointConfig() {
    max_keypoints = 450;
    keypoint_threshold = 0.004;
    remove_borders = 4;
    dla_core = 0;
    onnx_file = "";
    engine_file = "";
  }

  int max_keypoints;
  float keypoint_threshold;
  int remove_borders;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct ImgConfigs{
  // std::string dataroot;
  // std::string camera_config_path;
  ImgConfigs(){};
  ImgConfigs(std::string model_path) {
    // dataroot = "";
    // camera_config_path = "";
    model_dir = model_path;
    plnet_config.SetModelPath(model_path);
    point_matcher_config.onnx_file = model_dir + "/" + point_matcher_config.onnx_file;
    point_matcher_config.engine_file = model_dir + "/" + point_matcher_config.engine_file;
  }
  std::string model_dir;
  // std::string saving_dir;

  PLNetConfig plnet_config;
  SuperPointConfig superpoint_config;
  PointMatcherConfig point_matcher_config;


};


#endif