// This file is part of DVS-ROS - the RPG DVS ROS Package

#pragma once

#include "dvxplorer_mc_ceres/driver.h"

#include <nodelet/nodelet.h>

namespace dvxplorer_mc_ceres {

class DvxplorerMcCeresNodelet : public nodelet::Nodelet {
public:
	virtual void onInit();

private:
	dvxplorer_mc_ceres::DvxplorerMcCeres *driver_;
};

} // namespace dvxplorer_mc_ceres
