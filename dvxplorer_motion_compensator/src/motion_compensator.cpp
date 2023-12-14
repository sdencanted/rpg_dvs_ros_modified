#include "dvxplorer_motion_compensator/motion_compensator.h"

namespace dvxplorer_motion_compensator
{
class CudaAllocException : public std::exception
{
public:
  char* what()
  {
    return (char*)"Cuda Alloc Exception";
  }
};
DvxplorerMotionCompensator::DvxplorerMotionCompensator(const std::string& node_name, bool use_optitrack)
  : Node(node_name, rclcpp::NodeOptions{}.use_intra_process_comms(true)), use_optitrack_(use_optitrack)
{
  alloc_size = 10000;

  // if (!cudaAllocMapped(&exy_list_arr_, alloc_size * sizeof(uint16_t)))
  // {
  //   std::cout << "could not allocate cuda mem x" << std::endl;
  //   // throw CudaAllocException();
  // };
  // if (!cudaAllocMapped(&et_list_arr_, alloc_size * sizeof(float)))
  // {
  //   std::cout << "could not allocate cuda mem x" << std::endl;
  //   // throw CudaAllocException();
  // };

  // if (!cudaAllocMapped(&mat_data_, 640 * 480 * sizeof(int32_t)))
  // {
  //   std::cout << "could not allocate cuda mem mat_data" << std::endl;
  //   // throw CudaAllocException();
  // };
  // if (!cudaAllocMapped(&mat_data_rescaled_, 640 * 480 * sizeof(uint8_t)))
  // {
  //   std::cout << "could not allocate cuda mem mat_data_rescaled" << std::endl;
  //   // throw CudaAllocException();
  // };
  event_mag_struct_sub_ = create_subscription<dvs_msgs::msg::EventMagStruct>(
      (ns + "/eventMagStruct").c_str(), 1,
      std::bind(&DvxplorerMotionCompensator::callback, this, std::placeholders::_1));
  if (use_optitrack_)
  {
    vrpn_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/rotationrig/pose", 1, std::bind(&DvxplorerMotionCompensator::vrpn_callback, this, std::placeholders::_1));
  }
  image_pub_ = create_publisher<sensor_msgs::msg::Image>(ns + "/image", 10);
  uncompensated_image_pub_ = create_publisher<sensor_msgs::msg::Image>(ns + "/image_raw", 10);
  RCLCPP_INFO(this->get_logger(), "motion compensator initialized");
}
void DvxplorerMotionCompensator::vrpn_callback(geometry_msgs::msg::PoseStamped::UniquePtr msg)
{
  tf2::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
  // double roll, pitch, yaw;
  tf2::Matrix3x3 m(q);
  double unused;
  m.getRPY(unused, unused, theta_);
  RCLCPP_INFO(this->get_logger(), "theta %.3f",theta_);

  // RCLCPP_INFO(this->get_logger(),"rpy %.3f %.3f %.3f",roll,pitch,yaw);
  // theta_=yaw;
}
void DvxplorerMotionCompensator::callback(dvs_msgs::msg::EventMagStruct::UniquePtr msg)
{
  double theta=theta_;
  if (use_optitrack_)
  {
    if (prev_theta_ == -10)
    {
      prev_theta_ = theta;
      return;
    }
  }

  // RCLCPP_INFO(this->get_logger(),"callback start %3f %3f",msg->first_theta,msg->last_theta);
  result_ = sensor_msgs::msg::Image::UniquePtr(new sensor_msgs::msg::Image());
  result_->height = 480 * 2;
  result_->width = 640;
  // result_->encoding = "MONO8";
  result_->encoding = "8UC1";
  result_->step = 640 * sizeof(uint8_t);
  result_->data.resize(result_->height * result_->step, 0);
  float first_timestamp = msg->event_time.data.front();
  float duration = msg->event_time.data.back() - first_timestamp;
  auto event_size = msg->event_time.data.size();

  cudaDeviceSynchronize();

  if (!cudaAllocMapped(&exy_list_arr_, event_size * 2 * sizeof(uint16_t)))
  {
    std::cout << "could not allocate cuda mem x" << std::endl;
    // throw CudaAllocException();
  };
  if (!cudaAllocMapped(&et_list_arr_, event_size * sizeof(float)))
  {
    std::cout << "could not allocate cuda mem x" << std::endl;
    // throw CudaAllocException();
  };

  if (!cudaAllocMapped(&mat_data_, 640 * 480 * 2 * sizeof(int32_t)))
  {
    std::cout << "could not allocate cuda mem mat_data" << std::endl;
    // throw CudaAllocException();
  };
  if (!cudaAllocMapped(&mat_data_rescaled_, 640 * 480 * 2 * sizeof(uint8_t)))
  {
    std::cout << "could not allocate cuda mem mat_data_rescaled" << std::endl;
    // throw CudaAllocException();
  };
  std::fill_n(this->mat_data_, (640 * 480 * 2), 0);

  // if (this->alloc_size < event_size)
  // {
  //   CUDA(cudaFreeHost(this->exy_list_arr_));
  //   CUDA(cudaFreeHost(this->et_list_arr_));
  //   this->alloc_size = event_size * 2;
  //   if (!cudaAllocMapped(&this->exy_list_arr_, this->alloc_size* sizeof(uint16_t)))
  //   {
  //     std::cout << "could not allocate cuda mem x" << std::endl;
  //     // throw CudaAllocException();
  //   };
  //   if (!cudaAllocMapped(&this->et_list_arr_, this->alloc_size * sizeof(float)))
  //   {
  //     std::cout << "could not allocate cuda mem x" << std::endl;
  //     // throw CudaAllocException();
  //   };
  // }

  std::copy(msg->event_arr.data.begin(), msg->event_arr.data.end(), this->exy_list_arr_);
  std::copy(msg->event_time.data.begin(), msg->event_time.data.end(), this->et_list_arr_);
  if (use_optitrack_)
  {
    double theta_diff;
    if (theta < prev_theta_)
    {
      theta_diff = M_PI * 2 + theta - prev_theta_;
    }
    else
    {
      theta_diff = theta - prev_theta_;
    }
    // motionCompensate(this->exy_list_arr_, this->et_list_arr_, this->mat_data_, (int)event_size, first_timestamp,
    //                  duration, fx, fy, cx, cy, theta_diff);
    motionCompensate(this->exy_list_arr_, this->et_list_arr_, this->mat_data_, (int)event_size, first_timestamp,
                     duration, fx, fy, cx, cy, -1.0*2*M_PI*duration);
    RCLCPP_INFO(this->get_logger(), "motion compensation %.3f %.3f %.3f %.3f %.3f %.3f %.3f", first_timestamp, duration,
                theta, prev_theta_, theta_diff, msg->event_time.data.front(), msg->event_time.data.back());
  }
  else
  {
    motionCompensate(this->exy_list_arr_, this->et_list_arr_, this->mat_data_, (int)event_size, first_timestamp,
                     duration, fx, fy, cx, cy, msg->first_theta - msg->last_theta);
  }
  // rescale(this->mat_data_, this->mat_data_rescaled_, 40.0);
  // cudaDeviceSynchronize();
  // std::copy(this->mat_data_rescaled_, this->mat_data_rescaled_+480*640, this->result_->data.data());
  // this->image_pub_->publish(std::move(this->result_));

  // std::fill_n(this->mat_data_, (640 * 480), 0);
  accumulate(this->exy_list_arr_, this->mat_data_ + 640 * 480, (int)event_size);
  rescale(this->mat_data_, this->mat_data_rescaled_, 40.0);
  cudaDeviceSynchronize();
  std::copy(this->mat_data_rescaled_, this->mat_data_rescaled_ + 480 * 640 * 2, this->result_->data.data());
  // this->uncompensated_image_pub_->publish(std::move(this->result_));

  this->image_pub_->publish(std::move(this->result_));
  CUDA(cudaFreeHost(exy_list_arr_));
  CUDA(cudaFreeHost(et_list_arr_));
  CUDA(cudaFreeHost(mat_data_));
  CUDA(cudaFreeHost(mat_data_rescaled_));
  // RCLCPP_INFO(this->get_logger(),"published");

  this->result_.reset();
  
  if (use_optitrack_)
  {
    prev_theta_ = theta;
  }
}
}  // namespace dvxplorer_motion_compensator