#include "dvxplorer_motion_compensator/motion_compensator.h"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char *argv[]) {
	rclcpp::init(argc, argv);
	auto motion_compensator = std::make_shared<dvxplorer_motion_compensator::DvxplorerMotionCompensator>("dvxplorer_motion_compensator",true);
	rclcpp::spin(motion_compensator);
	return 0;
}
