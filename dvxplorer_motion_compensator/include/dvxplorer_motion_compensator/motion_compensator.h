// This file is part of DVS-ROS - the RPG DVS ROS Package

#pragma once

// #include <ros/ros.h>
#include <rclcpp/rclcpp.hpp>
#include <string>

// boost
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread_time.hpp>

// messages
#include <dvs_msgs/msg/event_mag_struct.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "dvxplorer_motion_compensator/cuda_compensator.h"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

namespace dvxplorer_motion_compensator
{
class DvxplorerMotionCompensator : public rclcpp::Node
{
public:
  explicit DvxplorerMotionCompensator(const std::string& node_name,bool use_optitrack=false);
  // DvxplorerMotionCompensator(const DvxplorerMotionCompensator&) = delete;
  // DvxplorerMotionCompensator& operator=(const DvxplorerMotionCompensator&) = delete;
  ~DvxplorerMotionCompensator()
  {
	// CUDA(cudaFreeHost(exy_list_arr_));
	// CUDA(cudaFreeHost(et_list_arr_));
	// CUDA(cudaFreeHost(mat_data_));
	// CUDA(cudaFreeHost(mat_data_rescaled_));
  };
  double fx{1.7731990397901502e+03},fy{1.7379582151842803e+03},cx{3.4163885089130241e+02},cy{4.4300233788931905e+02};
private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr uncompensated_image_pub_;
  std::string ns;
  std::atomic_bool running_;
  rclcpp::Subscription<dvs_msgs::msg::EventMagStruct>::SharedPtr event_mag_struct_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr vrpn_pose_sub_;
  void callback(dvs_msgs::msg::EventMagStruct::UniquePtr msg);
  void vrpn_callback(geometry_msgs::msg::PoseStamped::UniquePtr msg);
  double theta_=0;
  double prev_theta_=-10;
  uint16_t* exy_list_arr_ = NULL;
  float* et_list_arr_ = NULL;
  int32_t *mat_data_ = NULL;
  uint8_t *mat_data_rescaled_ = NULL;
  std::size_t alloc_size = 0;
  bool use_optitrack_=false;
  
  sensor_msgs::msg::Image::UniquePtr result_;
};

}  // namespace dvxplorer_motion_compensator