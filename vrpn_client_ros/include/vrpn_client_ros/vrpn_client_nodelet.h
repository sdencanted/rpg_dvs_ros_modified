// This file is part of DVS-ROS - the RPG DVS ROS Package

#pragma once

#include <nodelet/nodelet.h>

#include "vrpn_client_ros/vrpn_client_ros.h"

namespace vrpn_client_ros {

class VrpnClientRosNodelet : public nodelet::Nodelet {
public:
  virtual void onInit();

private:
  vrpn_client_ros::VrpnClientRos* client_;
};

}
