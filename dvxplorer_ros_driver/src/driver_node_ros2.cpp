// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_ros_driver/driver_ros2.h"

// #include <ros/ros.h>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char *argv[]) {
	// ros::init(argc, argv, "dvxplorer_ros_driver");
	rclcpp::init(argc, argv);

	// ros::NodeHandle nh;
	// ros::NodeHandle nh_private("~");
    // auto node = rclcpp::Node::make_shared("dvxplorer_ros_driver");
	auto driver = std::make_shared<dvxplorer_ros_driver::DvxplorerRosDriver>("dvxplorer_ros_driver");

	rclcpp::spin(driver);
	// rclcpp::spin();

	driver->dataStop();
	return 0;
}
