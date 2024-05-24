// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_mc_ceres/driver.h"

#include <ros/ros.h>

int main(int argc, char *argv[]) {
	ros::init(argc, argv, "dvxplorer_mc_ceres");

	ros::NodeHandle nh;
	ros::NodeHandle nh_private("~");

	dvxplorer_mc_ceres::DvxplorerMcCeres *driver = new dvxplorer_mc_ceres::DvxplorerMcCeres(nh, nh_private);

	ros::spin();

	return 0;
}
