// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_ros_driver/driver_ros2.h"
#include <dvxplorer_motion_compensator/motion_compensator.h>

// #include <ros/ros.h>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char *argv[]) {
	// ros::init(argc, argv, "dvxplorer_ros_driver");
	rclcpp::init(argc, argv);

	// ros::NodeHandle nh;
	// ros::NodeHandle nh_private("~");
    // auto node = rclcpp::Node::make_shared("dvxplorer_ros_driver");

	// make multithreaded executor
	rclcpp::executors::MultiThreadedExecutor executor;

	auto driver = std::make_shared<dvxplorer_ros_driver::DvxplorerRosDriver>("dvxplorer_ros_driver");
	auto motion_compensator = std::make_shared<dvxplorer_motion_compensator::DvxplorerMotionCompensator>("dvxplorer_motion_compensator");

	executor.add_node(driver);
	executor.add_node(motion_compensator);

	// wait for shutdown and then clean up
	executor.spin();
	driver->dataStop();
	RCLCPP_INFO(driver->get_logger(),"Exiting dvxplorer_ros_driver and dvxplorer_motion_compensator node");
	rclcpp::shutdown();
	// rclcpp::spin(driver);
	// rclcpp::spin();

	return 0;
}
