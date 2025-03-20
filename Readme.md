# LIR-LIVO
[LIR-LIVO: A Lightweight,Robust Lidar/Vision/Inertial Odometry with Illumination-Resilient Deep Features](https://arxiv.org/abs/2502.08676)

LIR-LIVO is a novel lidar-inertial-visual odometry system designed to deliver efficient and robust state estimation in challenging environments with degraded LiDAR signals and significant illumination variations. This system integrates advanced techniques, including LiDAR depth association, adaptive visual feature extraction using deep learning, and a lightweight visual-inertial architecture.

## Features
- **Illumination-Resilient Deep Features**: Incorporates deep learning-based SuperPoint and LightGlue algorithms for robust feature extraction and matching under challenging lighting conditions.
- **LiDAR Depth Association**: Efficiently associates LiDAR point clouds with visual features to enhance pose estimation.
- **Uniform Depth Distribution**: Uniform feature depth distribution is applied to improve the ESIKF of visual subsystem performance.
- **State-of-the-Art Performance**: Demonstrated superior accuracy, efficiency, and robustness compared to existing methods on multiple benchmark datasets.

## Datasets
LIR-LIVO has been extensively evaluated on several benchmark datasets:
- **NTU-VIRAL**: A dataset collected using drones, with ground truth provided by Leica Nova MS60.
- **Hilti'22**: Features indoor and outdoor environments with poor ambient lighting, ideal for testing robustness.
- **R3LIVE-Dataset**: Includes RGB cameras, LiVOX AVIA LiDAR sensors, and internal IMU data.
## Requirement
OpenCV 4.2
Eigen 3
TensorRT 8.6.1.6
CUDA 11.1
ROS noetic

## Thanks
Gratefully acknowledge the open-source projects Fast_LIVO[https://github.com/hku-mars/FAST-LIVO] and airSLAM[https://github.com/sair-lab/AirSLAM], whose excellent work and implementations have provided valuable inspiration and reference for the development of this program.