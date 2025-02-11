#include "IMG_Processing.h"



ImgProcess::ImgProcess( const ImgConfigs& configs, vk::AbstractCamera* cam){

  _point_matcher = std::shared_ptr<PointMatcher>(new PointMatcher(configs.point_matcher_config));
  _feature_detector = std::shared_ptr<FeatureDetector>(new FeatureDetector(configs.plnet_config));
  _cam = cam;
//   weight_function_.reset(new vk::robust_cost::HuberWeightFunction());
  reprojection_cov = 10;
  CLOUD_SIZE_USED = 10;
//   _feature_thread = std::thread(boost::bind(&ImgProcess::ExtractFeatureThread, this));
//   _tracking_thread = std::thread(boost::bind(&ImgProcess::TrackingThread, this));
}

Eigen::Matrix3d ImgProcess::SkewSymmetric(const Eigen::Vector3d &vector) {
    Eigen::Matrix3d mat;
    mat << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1), vector(0), 0;
    return mat;
}

double ImgProcess::getHuberLossScale( double reprojection_error, double outlier_threshold )
{
    // http://ceres-solver.org/nnls_modeling.html#lossfunction
    double scale = 1.0;
    if ( reprojection_error / outlier_threshold < 1.0 )
    {
        scale = 1.0;
    }
    else
    {
        scale = ( 2 * sqrt( reprojection_error ) / sqrt( outlier_threshold ) - 1.0 ) / reprojection_error;
    }
    return scale;
}

V3F ImgProcess::getpixel(cv::Mat img, V2D pc)
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]); 
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref-u_ref_i);
    const float subpix_v_ref = (v_ref-v_ref_i);
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    uint8_t* img_ptr = (uint8_t*) img.data + ((v_ref_i)* _cam->width()+ (u_ref_i))*3;
    float B = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[0+3] + w_ref_bl*img_ptr[_cam->width()*3] + w_ref_br*img_ptr[_cam->width()*3+0+3];
    float G = w_ref_tl*img_ptr[1] + w_ref_tr*img_ptr[1+3] + w_ref_bl*img_ptr[1+_cam->width()*3] + w_ref_br*img_ptr[_cam->width()*3+1+3];
    float R = w_ref_tl*img_ptr[2] + w_ref_tr*img_ptr[2+3] + w_ref_bl*img_ptr[2+_cam->width()*3] + w_ref_br*img_ptr[_cam->width()*3+2+3];
    V3F pixel(B,G,R);
    return pixel;
}

void ImgProcess::addInputData(const InputData& data ){
    
    cv::Mat image = data.image.clone();
    if(_cam->width()!=image.cols || _cam->height()!=image.rows)
    {
        // std::cout<<"Resize the img scale !!!"<<std::endl;
        double scale = double(_cam->width()) / double(image.cols);
        cv::resize(image,image,cv::Size(image.cols*scale,image.rows*scale),0,0,CV_INTER_LINEAR);
    }
    image_now = image.clone();
    cv::Mat img_gray;
    if(image.channels() != 1)
    {
        cv::cvtColor(image,image,cv::COLOR_BGR2GRAY);
    }
    _clouds.push_back(*(data.pointcloud));
    _clouds_state.push_back(*state);
    if(_clouds.size() < CLOUD_SIZE_USED)
    {
        return;
    }
    else//don't add when movement is too small
    {
        auto size = _clouds.size();
        V3D movement = _clouds_state[size-1].pos_end - _clouds_state[size-2].pos_end;
        V3D rotation = RotMtoEuler(_clouds_state[size-1].rot_end) - RotMtoEuler(_clouds_state[size-2].rot_end);
        if((movement.norm() < 0.1 && rotation.norm() < 0.01)|| _clouds.back().size() < 800)
        {
            
            _clouds.pop_back();
            _clouds_state.pop_back();
        }
    }
    
    // construct frame


    Eigen::Matrix<float, 259, Eigen::Dynamic> features; 
    static int counts = 0;
    static double time_d = 0;
    auto st= omp_get_wtime();
    _feature_detector->Detect(image, features);
    auto ed = omp_get_wtime();
    time_d += ed-st;
    counts++;
    std::cout<<"time_d = "<<time_d/counts<<std::endl;
    //   _feature_detector->removewithSemantic(image_left_semantic, left_features, left_lines);
    //   frame_type = _init ? FrameType::KeyFrame : FrameType::InitializationFrame;
      // SaveLineDetectionResult(image_left_rect, left_lines, _configs.saving_dir, std::to_string(frame->GetFrameId()));
    _features = features;
    _keypoints.clear();
    for(int i=0;i<features.cols();i++){
        _keypoints.push_back(Vector2d(features(1,i), features(2,i)));
    }
    if(getDownSamplePointCloud(*(data.pointcloud))){
        getPointByDSPointCloud();
    }
    else{
        _points.resize(_keypoints.size());
        _is_valid.resize(_keypoints.size(), false);
    }

    
    // cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    // for (int i=0;i< _keypoints.size();i++) {
    //     if(_is_valid[i]){
    //         cv::circle(image, cv::Point2f(_keypoints[i][0], _keypoints[i][1]), 3, cv::Scalar(255,0,0), -1);
    //     }
    //     else{
    //         cv::circle(image, cv::Point2f(_keypoints[i][0], _keypoints[i][1]), 3, cv::Scalar(0,0,255), -1); // 使用 -1 表示填充圆形
    //     }
    // }
    // cv::imwrite("/home/sjz/Desktop/livo_ws/src/livo_fast/src/img.png", image);
}

void ImgProcess::updateState()
{
    if(_cam_obs.size() >= frame_winsize){
        std::vector<M3D> Rwcs;
        std::vector<V3D> Twcs;
        StatesGroup state_tmp = *state;
        MD(2,3) Jdt,JdR;
        std::vector<std::vector<cv::DMatch>> matcheses;
        int total_num_obs = 0;
        bool EKF_end = false;

        for(int i=0;i<frame_winsize;i++){
            StatesGroup ref_state = _cam_obs[i].state;
            M3D Rwi0(ref_state.rot_end);
            V3D Twi0(ref_state.pos_end);
            M3D Rwc0 = Rwi0 * R_c2i;
            V3D Twc0 = Twi0 + Rwi0 * T_c2i;
            Rwcs.push_back(Rwc0);
            Twcs.push_back(Twc0);
            Eigen::Matrix<float, 259, Eigen::Dynamic> ref_features = _cam_obs[i].features;
            std::vector<PointType> ref_points = _cam_obs[i].points;
            std::vector<bool> is_valid = _cam_obs[i].is_valid;
            std::vector<cv::DMatch> matches;
            static int num_match = 0;
            static double total_time = 0.0;
            auto st = omp_get_wtime();
            _point_matcher->MatchingPoints(ref_features,_features,matches,true);
            auto ed = omp_get_wtime();
            num_match++;
            total_time += ed - st;
            std::cout << "\nMatching time: " << total_time/num_match<< std::endl;
            int num_opt = 0;
            double parallex = 0.0;
            for(int j=0;j<matches.size();j++){
                if(is_valid[matches[j].queryIdx]){
                    matches[num_opt] = matches[j];
                    parallex += (ref_features.block<2,1>(1,matches[j].queryIdx) - _features.block<2,1>(1,matches[j].trainIdx)).norm();
                    num_opt++;
                }
            }
            matches.resize(num_opt);
            if(num_opt == 0 || parallex / num_opt < 8.0)
            {
                matches.clear();
            }
            matcheses.push_back(matches);
            total_num_obs += matches.size();
        }
        if(total_num_obs == 0)
        {
            _cam_obs.pop_front();
            CamObs cam_obs;
            cam_obs.state = (*state);
            cam_obs.features = _features;
            cam_obs.image = image_now;
            cam_obs.points = _points;
            cam_obs.is_valid = _is_valid;
            _cam_obs.push_back(cam_obs);
            return;
        }
        auto ref_features = _cam_obs[0].features;
        std::vector<cv::KeyPoint> refps,currps;
        for(int i=0;i<ref_features.cols();i++)
        {
            refps.push_back(cv::KeyPoint(cv::Point2f(ref_features(1,i),ref_features(2,i)), 1.0f));
        }
        for(int i=0;i<_features.cols();i++)
        {
            currps.push_back(cv::KeyPoint(cv::Point2f(_features(1,i),_features(2,i)), 1.0f));
        }
        cv::Mat img_matches;
        cv::drawMatches(_cam_obs[0].image, refps, image_now, currps,matcheses[0] , img_matches, 
                    cv::Scalar::all(-1), cv::Scalar::all(-1), 
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // cv::imwrite("/home/sjz/Desktop/livo_ws/src/livo_fast/src/img.png", img_matches);
        img_draw = img_matches;

        z.resize(2 * total_num_obs);
        z.setZero();
        H_sub.resize(2 * total_num_obs,6);
        H_sub.setZero();
        H_T_H.setZero();
        G.setZero();
        double error_total = 0;
        double mean_error = 0;
        auto st = omp_get_wtime();
            
        for (int iteration =0 ; iteration < max_iter; iteration++){
            M3D Rwi1(state->rot_end);
            V3D Twi1(state->pos_end);
            M3D Rwc1 = Rwi1 * R_c2i;
            V3D Twc1 = Twi1 + Rwi1 * T_c2i;
            int outler =0 ;
            for(int match_it=0; match_it < matcheses.size();match_it++){
                auto matches = matcheses[match_it];
                int index = 0;
                for(int i=0;i<matches.size();i++){
                    Vector3d ref_point_3d { _cam_obs[match_it].points[matches[i].queryIdx].x , 
                                            _cam_obs[match_it].points[matches[i].queryIdx].y, 
                                            _cam_obs[match_it].points[matches[i].queryIdx].z}  ;
        
                    // auto uv_test = _cam->world2cam(ref_point_3d);
                    // Vector2d ref_UV = Vector2d(ref_features(1,matches[i].queryIdx), ref_features(2,matches[i].queryIdx));

                    // std::vector<cv::DMatch> matches_;
                    //         matches_.push_back(matches[i]);
                    //                 std::vector<cv::KeyPoint> refps,currps;
                    // for(int i=0;i<ref_features.cols();i++)
                    // {
                    //     refps.push_back(cv::KeyPoint(cv::Point2f(ref_features(1,i),ref_features(2,i)), 1.0f));
                    // }
                    // for(int i=0;i<_features.cols();i++)
                    // {
                    //     currps.push_back(cv::KeyPoint(cv::Point2f(_features(1,i),_features(2,i)), 1.0f));
                    // }
                    // cv::Mat img_matches;
                    // cv::drawMatches(_keyimages[refpos], refps, _keyimages.back(), currps,matches_ , img_matches, 
                    //             cv::Scalar::all(-1), cv::Scalar::all(-1), 
                    //             std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                    // cv::imwrite("/home/sjz/Desktop/livo_ws/src/livo_fast/src/img.png", img_matches);
                    M3D Rwc0 = Rwcs[match_it];
                    V3D Twc0 = Twcs[match_it];
                    Vector2d obs_UV = Vector2d(_features(1,matches[i].trainIdx), _features(2,matches[i].trainIdx));
                    Vector3d unit_obs = _cam->cam2world(obs_UV);
                    Vector2d obs = unit_obs.head(2) / unit_obs(2);
                    Vector3d point_w = Rwc0 * ref_point_3d + Twc0;
                    Vector3d point_c1 = Rwc1.transpose() * (point_w - Twc1);
                    Vector2d res = obs - point_c1.head(2) / point_c1(2);
                    error_total += res.norm() * weight_scale_unit;
                    // double weight = weight_function_->value(res.norm() * weight_scale_unit);
                    double weight = getHuberLossScale(res.norm() * weight_scale_unit,1);

                    // if(weight < 0.85)
                    // {
                    //     outler++;
                    //     weight *= 0.001;
                    // }
                    z.block<2,1>(2*index,0) =  res * weight * weight_scale_unit;
                    Jdt << -1 / point_c1(2), 0,  point_c1(0) / (point_c1(2) * point_c1(2)) ,
                            0, -1 / point_c1(2), point_c1(1) / (point_c1(2) * point_c1(2));
                    Jdt = Jdt * Rwc1.transpose();
                    JdR <<  1 / point_c1(2), 0, - point_c1(0) / (point_c1(2) * point_c1(2)),
                            0, 1 / point_c1(2), - point_c1(1) / (point_c1(2) * point_c1(2));
                    JdR = JdR * R_c2i.transpose() * SkewSymmetric( Rwi1.transpose() * (point_w - state->pos_end));
                    // std::cout<<"JdR: "<<JdR<<std::endl;
                    H_sub.block<2,3>(2*index,0) = JdR * weight * weight_scale_unit;
                    H_sub.block<2,3>(2*index,3) = Jdt * weight * weight_scale_unit;
                    index++;
                }
            }
            mean_error = error_total / (total_num_obs * 2);
            auto &&H_sub_T = H_sub.transpose();
            H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;

            MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / reprojection_cov).inverse()).inverse();
            auto &&HTz = H_sub_T * z;
            // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
            auto vec = (state_tmp) - (*state);
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
            auto solution = K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);
            // std::cout << "solution: " << solution.block<6,1>(0,0).transpose() << std::endl;
            (*state) += solution;
            auto &&rot_add = solution.block<3,1>(0,0);
            auto &&t_add   = solution.block<3,1>(3,0);
            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
            {
                EKF_end = true;
            }
            if(iteration  == max_iter - 1 || EKF_end){
                MD(DIM_STATE, DIM_STATE) I_STATE;
                I_STATE.setIdentity();
                state->cov = (I_STATE - G) * state->cov;
                break;
            }
        }
        auto ed = omp_get_wtime();
        std::cout << "\n solve time: " << ed - st<< std::endl;
        V3D movement = _cam_obs.back().state.pos_end - state->pos_end;
        V3D rotation = RotMtoEuler(_cam_obs.back().state.rot_end) - RotMtoEuler(state->rot_end);
        if(movement.norm() < 0.01 && rotation.norm() < 0.01)
        {
            return;
        }
        _cam_obs.pop_front();
        CamObs cam_obs;
        cam_obs.state = (*state);
        cam_obs.features = _features;
        cam_obs.points = _points;
        cam_obs.is_valid = _is_valid;
        cam_obs.image = image_now;
        _cam_obs.push_back(cam_obs);
    }
    else
    {
        if(_clouds.size() < CLOUD_SIZE_USED)
        {
            return;
        }
        CamObs cam_obs;
        cam_obs.state = (*state);
        cam_obs.features = _features;
        cam_obs.points = _points;
        cam_obs.is_valid = _is_valid;
        cam_obs.image = image_now;
        _cam_obs.push_back(cam_obs);
    }
}

void ImgProcess::outlierRejectF(const Eigen::Matrix<float, 259, Eigen::Dynamic>& ref_features,
                      const Eigen::Matrix<float, 259, Eigen::Dynamic>& cur_features,
                      const std::vector<cv::DMatch>& matches, std::vector<cv::DMatch>& matches_good)
{  
    std::vector<cv::Point2f> ref_points;
    std::vector<cv::Point2f> cur_points;

    for (const auto& match : matches) {
        int ref_idx = match.queryIdx;
        int cur_idx = match.trainIdx;

        // 获取参考图像和当前图像中匹配点的坐标
        float ref_x = ref_features(1, ref_idx);
        float ref_y = ref_features(2, ref_idx);
        float cur_x = cur_features(1, cur_idx);
        float cur_y = cur_features(2, cur_idx);

        ref_points.emplace_back(ref_x, ref_y);
        cur_points.emplace_back(cur_x, cur_y);
    }

    // 计算基本矩阵
    std::vector<uchar> inliers;
    cv::Mat fundamental_matrix = cv::findFundamentalMat(ref_points, cur_points, cv::FM_RANSAC, 3, 0.99, inliers);

    // 根据inliers筛选出好的匹配点
    matches_good.clear();
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inliers[i]) {
            matches_good.push_back(matches[i]);
        }
    }
}

bool ImgProcess::getDownSamplePointCloud(PointCloud &pointcloud) {
    
    PointCloud tmp_cloud;
    size_t cloud_size = _clouds.size();
    M3D R_c22w = state->rot_end * R_c2i;
    V3D T_c22w = state->pos_end + state->rot_end * T_c2i;
    for(int i = 0; i < cloud_size; i++){
        if(i==cloud_size - 1)
        {
            continue;
        }
        M3D R_c12w = _clouds_state[i].rot_end * R_c2i;
        V3D T_c12w = _clouds_state[i].pos_end + _clouds_state[i].rot_end * T_c2i;
        for(const auto &point : _clouds[i]){
            PointType pt;
            Vector3d p3d{point.x, point.y, point.z};
            p3d = R_c22w.transpose() * ( (R_c12w * p3d + T_c12w)  - T_c22w);
            pt.x = p3d[0]; pt.y = p3d[1]; pt.z = p3d[2];
            tmp_cloud.push_back(pt);
        }
    }
    for(auto & point : pointcloud){
        tmp_cloud.push_back(point);
    }

    // 点云数据投影降采样到一个距离图像, 仅保留最近距离的点, 剔除当前不可视的点, 并取前景点云
    // cv::Mat image = _keyimages.back();
    // cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    // for (size_t k = 0; k < tmp_cloud.size(); k++) {
    //     const auto &point = tmp_cloud[k];
    //     if (point.z < 0) {
    //         continue;
    //     }
    //     Vector3d p3d{point.x, point.y, point.z};
    //     Vector2d p2d = _cam->world2cam(p3d);
    //     cv::circle(image, cv::Point2f(p2d[0], p2d[1]), 1, cv::Scalar(255,0,0), -1);
    // }   

    // cv::imwrite("/home/sjz/Desktop/livo_ws/src/livo_fast/src/img.png", image);
    int width = _cam->width();
    int height = _cam->height();
    size_t point_size = tmp_cloud.size();
    // vector<float> range_image( width * height, FLT_MAX);
    // vector<PointType> points_valid(width * height);
    // for (size_t k = 0; k < point_size; k++) {
    //     const auto &point = tmp_cloud[k];

    //     // 不在图像中的点云
    //     if ((point.z < 0) || (fabs(atan2(point.x , point.z)) > horizontal_fov / 2.0 / 180.0 * M_PI) ||
    //         (fabs( atan2(point.y , point.z)) > vertical_fov / 2.0 / 180.0 * M_PI)) {
    //         continue;
    //     }

    //     // top -> bottom
    //     double row_angle =
    //         atan2(point.y,  point.z) + (M_PI * vertical_fov / 2.0 / 180.0);
    //     // right -> left
    //     double col_angle = atan2(point.x, point.z) + (M_PI * horizontal_fov / 2.0 / 180.0);
    //     // std::cout << "row_angle: " << row_angle / M_PI * 180.0 << " col_angle: " << col_angle / M_PI * 180.0 << std::endl;
    //     // 四舍五入
    //     int row = static_cast<int>(round(row_angle /  min_angular / M_PI * 180.0));
    //     int col = static_cast<int>(round(col_angle / min_angular / M_PI * 180.0));
    //     // std::cout << "row: " << row << " col: " << col << std::endl;
    //     if ((row < 0) || (col < 0) || (row >= height) || (col >= width)) {
    //         continue;
    //     }

    //     float distance = point.getVector3fMap().norm();
    //     int index      = col + row * width;
    //     if (distance < range_image[index]) {
    //         range_image[index]  = distance;
    //         points_valid[index] = tmp_cloud[k];
    //     }
    // }

    // 抽取降采样的点云投影到单位球相机坐标系

    unit_sphere_pointcloud_ = PointCloudPtr(new PointCloud);
    // for (int i = 0; i < 500; i++) {
    //     for (int j = 0; j < 500; j++) {
    //         int index = j + i * 500;
    //         if (range_image[index] != FLT_MAX) {

    //             PointType p;
    //             // 强度数据保存距离信息
    //             p.intensity        = points_valid[index].getVector3fMap().norm();
    //             p.getVector3fMap() = points_valid[index].getVector3fMap().normalized();

    //             unit_sphere_pointcloud_->push_back(p);
    //         }
    //     }
    // }
    for(int i =0;i<tmp_cloud.size();i++){
        const auto &point = tmp_cloud[i];
        if(tmp_cloud[i].z<0 ||fabs( atan2(point.x , point.z)) > horizontal_fov / 2.0 / 180.0 * M_PI ||
            fabs( atan2(point.y , point.z)) > vertical_fov / 2.0 / 180.0 * M_PI)
        {
            continue;
        }    
        PointType p;
        p.intensity        = tmp_cloud[i].getVector3fMap().norm();
        p.getVector3fMap() = tmp_cloud[i].getVector3fMap().normalized();
        unit_sphere_pointcloud_->push_back(p);
    }

    if(_clouds.size()>CLOUD_SIZE_USED)
    {
        _clouds.pop_front();
        _clouds_state.pop_front();
    }

    // cv::Mat image = image_now;
    // cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    // for (size_t k = 0; k < unit_sphere_pointcloud_->size(); k++) {
    //     const auto &point = (*unit_sphere_pointcloud_)[k];
    //     if (point.z < 0) {
    //         continue;
    //     }
    //     Vector3d p3d{point.x, point.y, point.z};
    //     Vector2d p2d = _cam->world2cam(p3d);
    //     cv::circle(image, cv::Point2f(p2d[0], p2d[1]), 1, cv::Scalar(255,0,0), -1);
    // }   

    // cv::imwrite("/home/sjz/Desktop/livo_ws/src/livo_fast/src/img_lidar.png", image);


    // 建立KD树
     if (unit_sphere_pointcloud_->empty()) {
        return false;
    }
    _kdtree = pcl::KdTreeFLANN<PointType>::Ptr(new pcl::KdTreeFLANN<PointType>());
    _kdtree->setInputCloud(unit_sphere_pointcloud_);

    return !unit_sphere_pointcloud_->empty();
}


void ImgProcess::getPointByDSPointCloud() {



    // 视觉特征投影到单位球相机坐标系

    // 参考帧的跟踪上的未三角化特征点

    PointCloud unit_sphere_features;
    // for (int i=0; i < _keypoints.size(); i++ ) {
    //     PointType point;
    //     Vector3d puc           = _cam->cam2world(_keypoints[i]);
    //     point.getVector3fMap() = puc.cast<float>();
    //     point.intensity        = 0;

    //     unit_sphere_features.push_back(point);
    // }
    for (const auto &pts2d : _keypoints) {
        PointType point;
        Vector3d puc           = _cam->cam2world(pts2d);
        point.getVector3fMap() = puc.cast<float>();
        point.intensity        = 0;

        unit_sphere_features.push_back(point);
    }
    // std::cout << "unit_sphere_features.size():" << unit_sphere_features.size() << std::endl;
    // img_draw = _keyimages.back();
    // cv::cvtColor(img_draw, img_draw, cv::COLOR_GRAY2BGR);
    _points.clear();
    _is_valid.clear();
    for (size_t k = 0; k < unit_sphere_features.size(); k++) {
        // 非当前参考帧, 跳过
        auto &point = unit_sphere_features.points[k];

        bool type = checkPointPlane(point);
    
        _points.push_back(point);
        _is_valid.push_back(type);
    }
    // distributePointsByDepth(_points, _is_valid, 50);
    // cv::imwrite("/home/sjz/Desktop/livo_ws/src/livo_fast/src/img.png", img_draw);
}

bool ImgProcess::checkPointPlane(PointType &point) {
    vector<int> k_indices(0);
    vector<float> k_sqr_distances(0);


    // 最近邻搜索
    _kdtree->nearestKSearch(point, 5, k_indices, k_sqr_distances);

    if (k_sqr_distances[4] > 0.005 || k_indices.size() < 5) {
        return false;
    }
    Vector3d pt3d(point.x, point.y, point.z);
    Vector2d pt2d = _cam->world2cam(pt3d);
    vector<Eigen::Vector3d> points;
    double ave_dis = 0;
    for (int k = 0; k < 5; k++) {
        const auto &pt = unit_sphere_pointcloud_->points[k_indices[k]];
        Vector3d pt_check(pt.x, pt.y, pt.z);
        Vector2d pt_uv = _cam->world2cam(pt_check);
        ave_dis += (pt_uv - pt2d).norm();
        points.emplace_back(pt.getVector3fMap().cast<double>() * pt.intensity);
    }
    ave_dis /= 5.0;
    if (ave_dis > 5) {
        return false;
    }
    // 构建正定方程求解平面法向量
    Eigen::Matrix<double, 5, 3> ma;
    Eigen::Matrix<double, 5, 1> mb = -Eigen::Matrix<double, 5, 1>::Ones();
    for (int k = 0; k < 5; k++) {
        ma.row(k) = points[k];
    }
    Eigen::Vector3d mx = ma.colPivHouseholderQr().solve(mb);

    // 归一化处理
    double norm_inverse = 1.0 / mx.norm();
    mx.normalize();

    // 校验平面
    bool isvalid = true;
    vector<double> threshold;
    for (int k = 0; k < 5; k++) {
        double sm = fabs(mx.dot(points[k]) + norm_inverse);
        threshold.push_back(sm);
        if (sm > 0.05) {
            isvalid = false;
            break;
        }
    }
    double depth_diff = fabs(unit_sphere_pointcloud_->points[k_indices[0]].intensity -
                             unit_sphere_pointcloud_->points[k_indices[4]].intensity);
    if (depth_diff > 0.1) {
        isvalid = false;
    }
    if (isvalid) {
        // 选择距离最远的点构建方程求解深度
        Eigen::Vector3f vr = points[4].cast<float>();
        Eigen::Vector3f vi = point.getVector3fMap();
        Eigen::Vector3f vn = mx.cast<float>();

        float t = vn.dot(vr) / vn.dot(vi);

        // 有效特征深度, nan比较总是返回false
        if (std::isnan(t) || (t < 0.001) || (t > 300)) {
            return false;
        }

        // 相机坐标系下的绝对坐标
        point.getVector3fMap() *= t;

        // 深度信息
        point.intensity = point.z;
        //  绘制关联点
        // for(int i=0;i<5;i++){
        //     auto ptuv = _cam->world2cam(points[i]);
        //     cv::circle(img_draw, cv::Point2f(ptuv.x(), ptuv.y()), 1, cv::Scalar(255,0, 0), 1);
        // }
        // cv::circle(img_draw, cv::Point2f(pt2d.x(), pt2d.y()), 1, cv::Scalar(0,0, 255), 1);
    }

    return isvalid;
}

void ImgProcess::distributePointsByDepth(const std::vector<PointType>& points, std::vector<bool>& is_valid, size_t target_count) {
    // 检查输入是否匹配
    if (points.size() != is_valid.size() || points.empty()) {
        return;
    }

    // 仅保留可用的点（_is_valid为true）
    std::vector<std::pair<PointType, size_t>> valid_points; // 保存点和其索引
    for (size_t i = 0; i < points.size(); ++i) {
        if (is_valid[i]) {
            valid_points.emplace_back(points[i], i);
        }
    }

    // 如果有效点数少于目标数量，直接返回
    if (valid_points.size() <= target_count) {
        return; // 保持原有状态
    }

    // 按照深度（intensity）排序
    std::sort(valid_points.begin(), valid_points.end(), [](const auto& a, const auto& b) {
        return a.first.intensity < b.first.intensity;
    });

    // 计算均匀间隔
    float step = static_cast<float>(valid_points.size() - 1) / (target_count - 1);

    // 选择均匀分布的点
    std::vector<size_t> selected_indices;
    for (size_t i = 0; i < target_count; ++i) {
        size_t index = static_cast<size_t>(std::round(i * step));
        selected_indices.push_back(valid_points[index].second);
    }

    // 将所有点设置为不可用
    std::fill(is_valid.begin(), is_valid.end(), false);

    // 将均匀分布的点设置为可用
    for (size_t idx : selected_indices) {
        is_valid[idx] = true;
    }
}
