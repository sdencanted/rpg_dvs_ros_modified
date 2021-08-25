// This file is part of DVS-ROS - the RPG DVS ROS Package

#include <pluginlib/class_list_macros.h>

#include "vrpn_client_ros/vrpn_client_nodelet.h"

// watch the capitalization carefully
PLUGINLIB_EXPORT_CLASS(vrpn_client_ros::VrpnClientRosNodelet, nodelet::Nodelet)

namespace vrpn_client_ros
{

void VrpnClientRosNodelet::onInit()
{
  client_ = new vrpn_client_ros::VrpnClientRos(getNodeHandle(), getPrivateNodeHandle());

  NODELET_INFO_STREAM("Initialized " <<  getName() << " nodelet.");
}

//#ifndef PLUGINLIB_EXPORT_CLASS
//PLUGINLIB_DECLARE_CLASS(vrpn_client_ros, VrpnClientRosNodelet, vrpn_client_ros::VrpnClientRosNodelet, nodelet::Nodelet);
//#else
//PLUGINLIB_EXPORT_CLASS(vrpn_client_ros::VrpnClientRosNodelet, nodelet::Nodelet);
//#endif



}
