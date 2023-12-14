// This file is part of DVS-ROS - the RPG DVS ROS Package

#pragma once

// #include <ros/ros.h>
#include <rclcpp/rclcpp.hpp>
#include <string>

// boost
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread_time.hpp>

// messages
#include <dvs_msgs/msg/event.hpp>
#include <dvs_msgs/msg/event_array.hpp>
#include <dvs_msgs/msg/event_mag_struct.hpp>
#include <dvs_msgs/msg/event_image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/empty.hpp>
#include <builtin_interfaces/msg/time.hpp>

// DVXplorer driver
#include <libcaer/devices/dvxplorer.h>
#include <libcaer/libcaer.h>

// dynamic reconfigure
// #include <dvxplorer_ros_driver/DVXplorer_ROS_DriverConfig.h>
// #include <dynamic_reconfigure/server.h>

// camera info manager
#include <camera_info_manager/camera_info_manager.hpp>
// #include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/msg/camera_info.hpp>

// std algorithm to filter hot pixels
# include <algorithm>

// arm neon simd
#include<arm_neon.h>


namespace dvxplorer_ros_driver{

class DvxplorerRosDriver: public rclcpp::Node {
public:
	explicit DvxplorerRosDriver(const std::string & node_name);
	DvxplorerRosDriver(const DvxplorerRosDriver&) = delete;
	DvxplorerRosDriver& operator=(const DvxplorerRosDriver&) = delete;
	~DvxplorerRosDriver();

	void dataStop();

	static void onDisconnectUSB(void *);

private:
	// rcl_interfaces::msg::SetParametersResult callback(std::vector<rclcpp::Parameter> parameters);
	rcl_interfaces::msg::SetParametersResult callback(std::vector<rclcpp::Parameter>);
	// void callback(dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig &config, uint32_t level);
	void readout();
	void resetTimestamps();
	void caerConnect();
	void caerConnect2();

	// ros::NodeHandle nh_;
	// ros::Publisher event_array_pub_;
	rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
    rclcpp::Publisher<dvs_msgs::msg::EventMagStruct>::SharedPtr event_struct_pub_;
    rclcpp::Publisher<dvs_msgs::msg::EventImage>::SharedPtr event_image_pub_;
    // rclcpp::Publisher<> event_size_pub_;
	rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
	caerDeviceHandle dvxplorer_handle_;

	std::string ns;

	std::atomic_bool running_;

	// boost::shared_ptr<dynamic_reconfigure::Server<dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig>> server_;
	// dynamic_reconfigure::Server<dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig>::CallbackType
	// 	dynamic_reconfigure_callback_;
	rclcpp::Subscription<builtin_interfaces::msg::Time>::SharedPtr reset_sub_;
	rclcpp::Publisher<builtin_interfaces::msg::Time>::SharedPtr reset_pub_;
	void resetTimestampsCallback(builtin_interfaces::msg::Time::UniquePtr msg);

	rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr imu_calibration_sub_;
	void imuCalibrationCallback(std_msgs::msg::Empty::UniquePtr );
	std::atomic_bool imu_calibration_running_;
	int imu_calibration_sample_size_;
	std::vector<sensor_msgs::msg::Imu> imu_calibration_samples_;
	sensor_msgs::msg::Imu bias;
	void updateImuBias();

	template<typename T>
	int sgn(T val) {
		return (T(0) < val) - (val < T(0));
	}

	boost::shared_ptr<boost::thread> readout_thread_;

	boost::posix_time::time_duration delta_;

	std::atomic_int streaming_rate_{100};
	std::atomic_int max_events_;

	std::shared_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
	rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

	struct caer_dvx_info dvxplorer_info_;
	bool master_;
	std::string device_id_;

	rclcpp::Time reset_time_;

	static constexpr double STANDARD_GRAVITY = 9.81;

	// void resetTimerCallback(const rclcpp::TimerEvent &te);
	void resetTimerCallback();
	// rclcpp::WallTimer<DvxplorerRosDriver::resetTimerCallback> timestamp_reset_timer_;
	rclcpp::TimerBase::SharedPtr timestamp_reset_timer_;

	// hot pixels
	// const std::vector<uint32_t> hot_pixels = {	27006, 38861, 51322, 74113, 94897, 
	// 											145521, 167170, 179678, 196177, 
	// 											199474, 203074, 209092, 237236, 
	// 											252650, 252651, 264467};
	const std::vector<uint32_t> hot_pixels = {	806, 8341, 16002, 30393, 43257,
												74921, 88410, 96358, 106617, 
												108714, 110874, 114732, 
												132316, 141970, 141971, 
												149227};

		
	
	
	
	float32x4_t gs1 = {0.002915024,0.013064233,0.021539279,0.013064233}; // Note: get actual value here
	float32x4_t gs2 = {0.013064233,0.058549832,0.096532353,0.058549832}; // 32bit floating point numbers have 
	float32x4_t gs3 = {0.021539279,0.096532353,0.159154943,0.096532353}; // between 6 and 7 digits of precision, 
																		 // regardless of exponent
	// ------ Magnetometer related
	float32_t prev_theta_;
	
};

} // namespace dvxplorer_ros_driver