// This file is part of DVS-ROS - the RPG DVS ROS Package

#pragma once

#include <ros/ros.h>
#include <string>

// boost
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread_time.hpp>

// messages
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <dvs_msgs/EventMagStruct.h>
#include <dvs_msgs/EventImage.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Time.h>

// DVXplorer driver
#include <libcaer/devices/dvxplorer.h>
#include <libcaer/libcaer.h>

// dynamic reconfigure
// #include <dvxplorer_ros_driver/DVXplorer_ROS_DriverConfig.h>
// #include <dynamic_reconfigure/server.h>

// camera info manager
#include <camera_info_manager/camera_info_manager.h>
#include <sensor_msgs/CameraInfo.h>

// std algorithm to filter hot pixels
# include <algorithm>

// arm neon simd
#include<arm_neon.h>

namespace dvxplorer_ros_driver {

class DvxplorerRosDriver {
public:
	DvxplorerRosDriver(ros::NodeHandle &nh, ros::NodeHandle nh_private);
	~DvxplorerRosDriver();

	void dataStop();

	static void onDisconnectUSB(void *);

private:
	void callback(dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig &config, uint32_t level);
	void readout();
	void resetTimestamps();
	void caerConnect();
	void caerConnect2();

	ros::NodeHandle nh_;
	// ros::Publisher event_array_pub_;
	ros::Publisher camera_info_pub_;
    ros::Publisher event_struct_pub_;
    ros::Publisher event_image_pub_;
    ros::Publisher event_size_pub_;
	ros::Publisher imu_pub_;
	caerDeviceHandle dvxplorer_handle_;

	std::string ns;

	std::atomic_bool running_;

	// boost::shared_ptr<dynamic_reconfigure::Server<dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig>> server_;
	// dynamic_reconfigure::Server<dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig>::CallbackType
	// 	dynamic_reconfigure_callback_;

	ros::Subscriber reset_sub_;
	ros::Publisher reset_pub_;
	void resetTimestampsCallback(const std_msgs::Time::ConstPtr &msg);

	ros::Subscriber imu_calibration_sub_;
	void imuCalibrationCallback(const std_msgs::Empty::ConstPtr &msg);
	std::atomic_bool imu_calibration_running_;
	int imu_calibration_sample_size_;
	std::vector<sensor_msgs::Imu> imu_calibration_samples_;
	sensor_msgs::Imu bias;
	void updateImuBias();

	template<typename T>
	int sgn(T val) {
		return (T(0) < val) - (val < T(0));
	}

	boost::shared_ptr<boost::thread> readout_thread_;

	boost::posix_time::time_duration delta_;

	std::atomic_int streaming_rate_;
	std::atomic_int max_events_;

	std::shared_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;

	struct caer_dvx_info dvxplorer_info_;
	bool master_;
	std::string device_id_;

	ros::Time reset_time_;

	static constexpr double STANDARD_GRAVITY = 9.81;

	ros::Timer timestamp_reset_timer_;
	void resetTimerCallback(const ros::TimerEvent &te);

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
	
};

} // namespace dvxplorer_ros_driver