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
#include <dvs_msgs/EventStruct.h>
#include <dvs_msgs/EventImage.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Time.h>

// DVXplorer driver
#include <libcaer/devices/dvxplorer.h>
#include <libcaer/libcaer.h>

// dynamic reconfigure
#include <dvxplorer_ros_driver/DVXplorer_ROS_DriverConfig.h>
#include <dynamic_reconfigure/server.h>

// camera info manager
#include <camera_info_manager/camera_info_manager.h>
#include <sensor_msgs/CameraInfo.h>

// std algorithm to filter hot pixels
#include <algorithm>

// arm neon simd
#include <arm_neon.h>

namespace dvxplorer_ros_driver
{

	class DvxplorerRosDriver
	{
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

		boost::shared_ptr<dynamic_reconfigure::Server<dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig>> server_;
		dynamic_reconfigure::Server<dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig>::CallbackType
			dynamic_reconfigure_callback_;

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

		template <typename T>
		int sgn(T val)
		{
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
		const std::vector<uint32_t> hot_pixels = {123,
2862,
4490,
6327,
6898,
6990,
8156,
8962,
9991,
10812,
11525,
12844,
13657,
13691,
14181,
14654,
16251,
17362,
17390,
18341,
18826,
19498,
20528,
21451,
22181,
22191,
22223,
25042,
25962,
26289,
28582,
28698,
29094,
29119,
29714,
29773,
31178,
32578,
32608,
34668,
34970,
35307,
38852,
38860,
39013,
39846,
39986,
40528,
40970,
41367,
41792,
42085,
42696,
43736,
44048,
45384,
46062,
46340,
47227,
47770,
48087,
48386,
48850,
49317,
49482,
50097,
50556,
54413,
54470,
54879,
55050,
56106,
58060,
58063,
59174,
61584,
62702,
63081,
65381,
65836,
66239,
66308,
69557,
69561,
72078,
72215,
72993,
73103,
74363,
75307,
76707,
76799,
77018,
77943,
79543,
80035,
81080,
81640,
83383,
83680,
84467,
85580,
85650,
86504,
87216,
89025,
90470,
91692,
92496,
92655,
93392,
94060,
95408,
96685,
96711,
98296,
99202,
99923,
100381,
101486,
105963,
107077,
108052,
108185,
108437,
109599,
110696,
110734,
111486,
116059,
116566,
117266,
119053,
120342,
120601,
121674,
122082,
123152,
123236,
123380,
125422,
126401,
126927,
128185,
128819,
128900,
129780,
130544,
131316,
131608,
136315,
136847,
137490,
138454,
141164,
143093,
144479,
144780,
145420,
146192,
147398,
147609,
149978,
150314,
150561,
151884,
152437,
153306,
154550,
154630,
158872


		};

		float32x4_t gs1 = {0.002915024, 0.013064233, 0.021539279, 0.013064233}; // Note: get actual value here
		float32x4_t gs2 = {0.013064233, 0.058549832, 0.096532353, 0.058549832}; // 32bit floating point numbers have
		float32x4_t gs3 = {0.021539279, 0.096532353, 0.159154943, 0.096532353}; // between 6 and 7 digits of precision,
																				// regardless of exponent
	};

} // namespace dvxplorer_ros_driver