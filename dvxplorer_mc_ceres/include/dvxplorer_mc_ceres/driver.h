// This file is part of DVS-ROS - the RPG DVS ROS Package

#pragma once

#include <ros/ros.h>
#include <string>

#include "ceres/ceres.h"
#include "ceres/numeric_diff_options.h"
// messages
#include <dvs_msgs/EventStruct.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/CompressedImage.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Time.h>
#include <turbojpeg.h>



// camera info manager
#include <camera_info_manager/camera_info_manager.h>
#include <sensor_msgs/CameraInfo.h>

#include "dvxplorer_mc_ceres/mc_gradient.h"
struct TjhandleDeleter
{
  void operator()(tjhandle * handle)
  {
    if (handle) {
      tjDestroy(*handle);
      delete handle;
    }
  }
};

using TjhandleUniquePtr = std::unique_ptr<tjhandle, TjhandleDeleter>;
inline TjhandleUniquePtr makeTjhandleUniquePtr()
{
  TjhandleUniquePtr ptr(new tjhandle, TjhandleDeleter());
  *ptr = tjInitCompress();
  return ptr;
}
namespace dvxplorer_mc_ceres {

class DvxplorerMcCeres {
public:
	DvxplorerMcCeres(ros::NodeHandle &nh, ros::NodeHandle nh_private);
	~DvxplorerMcCeres();

	void dataStop();

	static void onDisconnectUSB(void *);

private:

	ceres::LineSearchDirectionType line_search_direction_type_ = ceres::LBFGS;
    ceres::LineSearchType line_search_type_ = ceres::WOLFE;
    ceres::NonlinearConjugateGradientType nonlinear_conjugate_gradient_type_ = ceres::FLETCHER_REEVES;
    ceres::GradientProblemSolver::Options options_;
	std::shared_ptr<ceres::GradientProblem> problem_;
	std::shared_ptr<McGradient> mc_gr_;
	ros::NodeHandle nh_;
    ros::Subscriber event_struct_sub_;
	ros::Publisher compensated_image_pub_;
	std::string ns;
	void eventStructCallback(const dvs_msgs::EventStruct::ConstPtr &msg);	
    int height_=400;
    int width_=400;
    float fx_ = 1.7904096255342997e+03, fy_ = 1.7822557654303025e+03, cx_ = (3.2002555821529580e+02)-244, cy_ = (2.3647053629109917e+02)-84; // dvx micro
	double rotations_[3]={0.001};
	
	ceres::GradientProblemSolver::Summary summary_;
	TjhandleUniquePtr tjhandle_;
  int packet_count_=0;
};

} // namespace dvxplorer_mc_ceres