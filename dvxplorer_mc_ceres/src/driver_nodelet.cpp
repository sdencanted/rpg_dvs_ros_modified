// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_mc_ceres/driver_nodelet.h"

#include <pluginlib/class_list_macros.h>

namespace dvxplorer_mc_ceres {

void DvxplorerMcCeresNodelet::onInit() {
	driver_ = new dvxplorer_mc_ceres::DvxplorerMcCeres(getNodeHandle(), getPrivateNodeHandle());

	NODELET_INFO_STREAM("Initialized " << getName() << " nodelet.");
}

#ifndef PLUGINLIB_EXPORT_CLASS
PLUGINLIB_DECLARE_CLASS(
	dvxplorer_mc_ceres, DvxplorerMcCeresNodelet, dvxplorer_mc_ceres::DvxplorerMcCeresNodelet, nodelet::Nodelet);
#else
PLUGINLIB_EXPORT_CLASS(dvxplorer_mc_ceres::DvxplorerMcCeresNodelet, nodelet::Nodelet);
#endif

} // namespace dvxplorer_mc_ceres
