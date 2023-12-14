// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_ros_driver/driver_ros2.h"

#include <std_msgs/msg/int32.hpp>

#include <omp.h>

extern "C" {
#include "dvxplorer_ros_driver/rm3100_spi_userspace.h"
}

namespace dvxplorer_ros_driver
{
// DvxplorerRosDriver::DvxplorerRosDriver(const rclcpp::NodeOptions & options,const  rclcpp::NodeOptions &
// options_private) : nh_(nh), imu_calibration_running_(false) {
DvxplorerRosDriver::DvxplorerRosDriver(const std::string& node_name)
  : Node(node_name, rclcpp::NodeOptions{}.use_intra_process_comms(true)), imu_calibration_running_(false)
{
  // load parameters
  this->get_parameter_or<std::string>("serial_number", device_id_, "");
  this->get_parameter_or<bool>("master", master_, true);
  double reset_timestamps_delay;
  this->get_parameter_or<double>("reset_timestamps_delay", reset_timestamps_delay, -1.0);
  this->get_parameter_or<int>("imu_calibration_sample_size", imu_calibration_sample_size_, 1000);

  // initialize IMU bias
  this->get_parameter_or<double>("imu_bias/ax", bias.linear_acceleration.x, 0.0);
  this->get_parameter_or<double>("imu_bias/ay", bias.linear_acceleration.y, 0.0);
  this->get_parameter_or<double>("imu_bias/az", bias.linear_acceleration.z, 0.0);
  this->get_parameter_or<double>("imu_bias/wx", bias.angular_velocity.x, 0.0);
  this->get_parameter_or<double>("imu_bias/wy", bias.angular_velocity.y, 0.0);
  this->get_parameter_or<double>("imu_bias/wz", bias.angular_velocity.z, 0.0);

  // RM3100 mag
  start_mag();
  start_drdy();
  int revid_id = get_revid_id();
  printf("REVID(correct ID is 22): %X\n", revid_id);
  uint16_t cycle_count = 50;
//   uint16_t cycle_count = 200;
  change_cycle_count(cycle_count);
  set_continuous_measurement(false);
  uint8_t tmrc_value = 0x92;
//   uint8_t tmrc_value = 0x93;
  set_tmrc(tmrc_value);
  set_continuous_measurement(true);
  int read_cycle_count = get_cycle_count();
  printf("Cycle count: %d\n", read_cycle_count);

  // set namespace if still global
  ns = get_namespace();
  if (ns.find("/") != std::string::npos)
  {
	ns = ns.substr(ns.find("/") + 1);
  }

  RCLCPP_INFO(this->get_logger(), "namespace is [%s]", ns.c_str());

  if (ns == "/")
  {
	ns = "dvs";
  }

  event_struct_pub_ = create_publisher<dvs_msgs::msg::EventMagStruct>(ns + "/eventMagStruct", 10);
  event_image_pub_ = create_publisher<dvs_msgs::msg::EventImage>(ns + "/eventImage", 10);
  // event_size_pub_ = create_publisher<std_msgs::Int32>(ns+ "/eventSize",10);
  // event_array_pub_ = create_publisher<dvs_msgs::EventArray>(ns + "/events", 10);
  camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(ns + "/camera_info", 1);
  imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(ns + "/imu", 10);

  imu_calibration_sub_ = create_subscription<std_msgs::msg::Empty>(
	  (ns + "/calibrate_imu").c_str(), 1,
	  std::bind(&DvxplorerRosDriver::imuCalibrationCallback, this, std::placeholders::_1));
  // reset timestamps is publisher as master, subscriber as slave
  if (master_)
  {
	reset_pub_ = create_publisher<builtin_interfaces::msg::Time>((ns + "/reset_timestamps").c_str(), 1);
  }
  else
  {
	reset_sub_ = create_subscription<builtin_interfaces::msg::Time>(
		(ns + "/reset_timestamps").c_str(), 1,
		std::bind(&DvxplorerRosDriver::resetTimestampsCallback, this, std::placeholders::_1));
	// = nh_.subscribe((ns + "/reset_timestamps").c_str(), 1, &DvxplorerRosDriver::resetTimestampsCallback, this);
  }

  // Open device.
  caerConnect();

  // Dynamic reconfigure
  param_callback_handle_ =
	  add_on_set_parameters_callback(std::bind(&DvxplorerRosDriver::callback, this, std::placeholders::_1));
  // Will call callback, which will pass stored config to device.
  // dynamic_reconfigure_callback_ = boost::bind(&DvxplorerRosDriver::callback, this, _1, _2);
  // server_.reset(new dynamic_reconfigure::Server<dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig>(options_private));
  // server_->setCallback(dynamic_reconfigure_callback_);

  caerConnect2();

  // imu_calibration_sub_
  // =create_subscription<std_msgs::msg::Empty>((ns + "/calibrate_imu").c_str(), 1,
  // std::bind(&DvxplorerRosDriver::imuCalibrationCallback,this,std::placeholders::_1)); = nh_.subscribe((ns +
  // "/calibrate_imu").c_str(), 1, &DvxplorerRosDriver::imuCalibrationCallback, this);

  // start timer to reset timestamps for synchronization
  if (reset_timestamps_delay > 0.0)
  {
	// auto my_callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
	timestamp_reset_timer_
		// =create_wall_timer<int64_t,std::milli,std::function<void()>>(rclcpp::Duration(reset_timestamps_delay),
		// std::bind(&DvxplorerRosDriver::resetTimerCallback,this), my_callback_group);
		= create_wall_timer(std::chrono::nanoseconds(long(reset_timestamps_delay * 1e9)),
							[this]() -> void { resetTimerCallback(); });
	// = nh_.createTimer(rclcpp::Duration(reset_timestamps_delay), &DvxplorerRosDriver::resetTimerCallback, this);
	RCLCPP_INFO(this->get_logger(),
				"Started timer to reset timestamps on master DVS for synchronization (delay=%3.2fs).",
				reset_timestamps_delay);
  }
}

DvxplorerRosDriver::~DvxplorerRosDriver()
{
  end_mag();
  end_drdy();
  if (running_)
  {
	RCLCPP_INFO(this->get_logger(), "shutting down threads");
	running_ = false;
	readout_thread_->join();
	RCLCPP_INFO(this->get_logger(), "threads stopped");
	caerLog(CAER_LOG_ERROR, "destructor", "data stop now");
	caerDeviceDataStop(dvxplorer_handle_);
	caerDeviceClose(&dvxplorer_handle_);
  }
}

void DvxplorerRosDriver::dataStop()
{
  caerLog(CAER_LOG_INFO, "Exiting from driver node", "executing data stop");
  RCLCPP_INFO(this->get_logger(), "Exiting from driver node, executing data stop");
  caerDeviceDataStop(dvxplorer_handle_);
  caerDeviceClose(&dvxplorer_handle_);
}

void DvxplorerRosDriver::caerConnect()
{
  // start driver
  bool device_is_running = false;
  while (!device_is_running)
  {
	const char* serial_number_restrict = (device_id_.empty()) ? nullptr : device_id_.c_str();

	if (serial_number_restrict)
	{
	  RCLCPP_WARN(this->get_logger(), "Requested serial number: %s", device_id_.c_str());
	}

	dvxplorer_handle_ = caerDeviceOpen(1, CAER_DEVICE_DVXPLORER, 0, 0, serial_number_restrict);

	// was opening successful?
	device_is_running = (dvxplorer_handle_ != nullptr);

	if (!device_is_running)
	{
	  RCLCPP_WARN(this->get_logger(), "Could not find DVXplorer. Will retry every second.");
	  // rclcpp::Duration(1.0).sleep();
	  // rclcpp::Rate r(std::chrono::nanoseconds(1e9));
	  // r.sleep();
	  rclcpp::sleep_for(std::chrono::nanoseconds(long(1e9)));

	  // rclcpp::spin_some(this);
	  // rclcpp::spin_some(this->get_node_base_interface());
	}

	if (!rclcpp::ok())
	{
	  return;
	}
  }

  dvxplorer_info_ = caerDVXplorerInfoGet(dvxplorer_handle_);
  device_id_ = "DVXplorer-" + std::string(dvxplorer_info_.deviceSerialNumber);

  RCLCPP_INFO(this->get_logger(), "%s --- ID: %d, Master: %d, DVS X: %d, DVS Y: %d, Firmware: %d, Logic: %d.\n",
			  dvxplorer_info_.deviceString, dvxplorer_info_.deviceID, dvxplorer_info_.deviceIsMaster,
			  dvxplorer_info_.dvsSizeX, dvxplorer_info_.dvsSizeY, dvxplorer_info_.firmwareVersion,
			  dvxplorer_info_.logicVersion);

  if (master_ && !dvxplorer_info_.deviceIsMaster)
  {
	RCLCPP_WARN(this->get_logger(), "Device %s should be master, but is not!", device_id_.c_str());
  }

  // Send the default configuration before using the device.
  // No configuration is sent automatically!
  caerDeviceSendDefaultConfig(dvxplorer_handle_);

  // // spawn threads
  // running_ = true;
  // readout_thread_
  // 	= boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&DvxplorerRosDriver::readout, this)));

  // // camera info handling
  // ros::NodeHandle nh_ns(ns);
  // camera_info_manager_.reset(new camera_info_manager::CameraInfoManager(this, device_id_));

  // // initialize timestamps
  // resetTimestamps();
}

void DvxplorerRosDriver::caerConnect2()
{
  // Only spawn threads after Dynamic reconfigure started.
  // spawn threads
  running_ = true;
  readout_thread_ =
	  boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&DvxplorerRosDriver::readout, this)));

  // camera info handling
  // ros::NodeHandle nh_ns(ns);
  // rclcpp::Node nh_ns(ns+"_camera_info_manager");
  camera_info_manager_.reset(new camera_info_manager::CameraInfoManager(this, device_id_));

  // initialize timestamps
  resetTimestamps();
}

void DvxplorerRosDriver::onDisconnectUSB(void* driver)
{
  RCLCPP_ERROR(rclcpp::get_logger("dvxplorer_ros_driver"), "USB connection lost with DVS !");
  static_cast<dvxplorer_ros_driver::DvxplorerRosDriver*>(driver)->caerConnect();
}

void DvxplorerRosDriver::resetTimestamps()
{
  caerDeviceConfigSet(dvxplorer_handle_, DVX_MUX, DVX_MUX_TIMESTAMP_RESET, 1);
  reset_time_ = get_clock()->now();

  RCLCPP_INFO(get_logger(), "Reset timestamps on %s to %.9f.", device_id_.c_str(), reset_time_.seconds());

  // if master, publish reset time to slaves
  if (master_)
  {
	auto reset_msg = builtin_interfaces::msg::Time();
	reset_msg.set__nanosec(reset_msg.nanosec);
	reset_msg.set__sec(reset_msg.sec);
	// reset_msg.data = reset_time_;
	reset_pub_->publish(reset_msg);
  }
}

void DvxplorerRosDriver::resetTimestampsCallback(builtin_interfaces::msg::Time::UniquePtr msg)
{
  // if slave, only adjust offset time
  if (!dvxplorer_info_.deviceIsMaster)
  {
	RCLCPP_INFO(this->get_logger(), "Adapting reset time of master on slave %s.", device_id_.c_str());
	reset_time_ = rclcpp::Time(msg->sec, msg->nanosec);
	// reset_time_ = msg->data;
  }
  // if master, or not single camera configuration, just reset timestamps
  else
  {
	resetTimestamps();
  }
}

void DvxplorerRosDriver::imuCalibrationCallback(std_msgs::msg::Empty::UniquePtr)
{
  RCLCPP_INFO(this->get_logger(), "Starting IMU calibration with %d samples...", (int)imu_calibration_sample_size_);
  imu_calibration_running_ = true;
  imu_calibration_samples_.clear();
}

void DvxplorerRosDriver::updateImuBias()
{
  bias.linear_acceleration.x = 0.0;
  bias.linear_acceleration.y = 0.0;
  bias.linear_acceleration.z = 0.0;
  bias.angular_velocity.x = 0.0;
  bias.angular_velocity.y = 0.0;
  bias.angular_velocity.z = 0.0;

  for (const auto& m : imu_calibration_samples_)
  {
	bias.linear_acceleration.x += m.linear_acceleration.x;
	bias.linear_acceleration.y += m.linear_acceleration.y;
	bias.linear_acceleration.z += m.linear_acceleration.z;
	bias.angular_velocity.x += m.angular_velocity.x;
	bias.angular_velocity.y += m.angular_velocity.y;
	bias.angular_velocity.z += m.angular_velocity.z;
  }

  bias.linear_acceleration.x /= (double)imu_calibration_samples_.size();
  bias.linear_acceleration.y /= (double)imu_calibration_samples_.size();
  bias.linear_acceleration.z /= (double)imu_calibration_samples_.size();
  bias.linear_acceleration.z -= STANDARD_GRAVITY * sgn(bias.linear_acceleration.z);

  bias.angular_velocity.x /= (double)imu_calibration_samples_.size();
  bias.angular_velocity.y /= (double)imu_calibration_samples_.size();
  bias.angular_velocity.z /= (double)imu_calibration_samples_.size();

  RCLCPP_INFO(this->get_logger(), "IMU calibration done.");
  RCLCPP_INFO(this->get_logger(), "Acceleration biases: %1.5f %1.5f %1.5f [m/s^2]", bias.linear_acceleration.x,
			  bias.linear_acceleration.y, bias.linear_acceleration.z);
  RCLCPP_INFO(this->get_logger(), "Gyroscope biases: %1.5f %1.5f %1.5f [rad/s]", bias.angular_velocity.x,
			  bias.angular_velocity.y, bias.angular_velocity.z);
}

// void DvxplorerRosDriver::resetTimerCallback(const ros::TimerEvent &te) {
void DvxplorerRosDriver::resetTimerCallback()
{
  timestamp_reset_timer_.reset();
  resetTimestamps();
}

// void DvxplorerRosDriver::callback(dvxplorer_ros_driver::DVXplorer_ROS_DriverConfig &config, uint32_t level) {
// 	// All changes have 'level' set.
// 	if (level == 0) {
// 		return;
// 	}

// 	// DVS control.
// 	if (level & (0x01 << 0)) {
// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_DVS, DVX_DVS_RUN, config.dvs_enabled);
// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_DVS_CHIP_BIAS, DVX_DVS_CHIP_BIAS_SIMPLE, config.bias_sensitivity);
// 	}

// 	// Subsampling.
// 	if (level & (0x01 << 1)) {
// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_DVS_CHIP, DVX_DVS_CHIP_SUBSAMPLE_ENABLE, config.subsample_enable);
// 		caerDeviceConfigSet(
// 			dvxplorer_handle_, DVX_DVS_CHIP, DVX_DVS_CHIP_SUBSAMPLE_HORIZONTAL, config.subsample_horizontal);
// 		caerDeviceConfigSet(
// 			dvxplorer_handle_, DVX_DVS_CHIP, DVX_DVS_CHIP_SUBSAMPLE_VERTICAL, config.subsample_vertical);
// 	}

// 	// Polarity control.
// 	if (level & (0x01 << 2)) {
// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_DVS_CHIP, DVX_DVS_CHIP_EVENT_FLATTEN, config.polarity_flatten);
// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_DVS_CHIP, DVX_DVS_CHIP_EVENT_ON_ONLY, config.polarity_on_only);
// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_DVS_CHIP, DVX_DVS_CHIP_EVENT_OFF_ONLY, config.polarity_off_only);
// 	}

// 	// DVS Region Of Interest.
// 	if (level & (0x01 << 3)) {
// 		auto a = caerDeviceConfigSet(dvxplorer_handle_, DVX_DVS_CHIP_CROPPER, DVX_DVS_CHIP_CROPPER_ENABLE,
// config.roi_enabled); 		caerDeviceConfigSet( 			dvxplorer_handle_, DVX_DVS_CHIP_CROPPER,
// DVX_DVS_CHIP_CROPPER_X_START_ADDRESS, config.roi_start_column); 		caerDeviceConfigSet( 			dvxplorer_handle_,
// DVX_DVS_CHIP_CROPPER, DVX_DVS_CHIP_CROPPER_X_END_ADDRESS, config.roi_end_column); 		caerDeviceConfigSet(
// 			dvxplorer_handle_, DVX_DVS_CHIP_CROPPER, DVX_DVS_CHIP_CROPPER_Y_START_ADDRESS, config.roi_start_row);
// 		caerDeviceConfigSet(
// 			dvxplorer_handle_, DVX_DVS_CHIP_CROPPER, DVX_DVS_CHIP_CROPPER_Y_END_ADDRESS, config.roi_end_row);
// 		RCLCPP_INFO(this->get_logger(),"ROI config set: %s",  a ? "true" : "false");
// 	}

// 	// Inertial Measurement Unit.
// 	if (level & (0x01 << 4)) {
// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_IMU, DVX_IMU_RUN_ACCELEROMETER, config.imu_enabled);
// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_IMU, DVX_IMU_RUN_GYROSCOPE, config.imu_enabled);

// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_IMU, DVX_IMU_ACCEL_RANGE, config.imu_acc_scale);
// 		caerDeviceConfigSet(dvxplorer_handle_, DVX_IMU, DVX_IMU_GYRO_RANGE, config.imu_gyro_scale);
// 	}

// 	// Streaming rate changes.
// 	if (level & (0x01 << 5)) {
// 		if (config.streaming_rate > 0) {
// 			delta_ = boost::posix_time::microseconds(long(1e6 / config.streaming_rate));
// 		}

// 		streaming_rate_ = config.streaming_rate;
// 		max_events_     = config.max_events;
// 	}
// }

// rcl_interfaces::msg::SetParametersResult DvxplorerRosDriver::callback(std::vector<rclcpp::Parameter> parameters){
rcl_interfaces::msg::SetParametersResult DvxplorerRosDriver::callback(std::vector<rclcpp::Parameter>)
{
  RCLCPP_WARN(this->get_logger(), "parameter callback not implemented!");
  auto result = rcl_interfaces::msg::SetParametersResult();
  result.successful = false;
  return result;
}

void DvxplorerRosDriver::readout()
{
  // std::vector<dvs::Event> events;

  caerDeviceConfigSet(dvxplorer_handle_, CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);
  caerDeviceDataStart(dvxplorer_handle_, nullptr, nullptr, nullptr, &DvxplorerRosDriver::onDisconnectUSB, this);

  // boost::posix_time::ptime next_send_time = boost::posix_time::microsec_clock::local_time();

  // dvs_msgs::EventArrayPtr event_array_msg;
  dvs_msgs::msg::EventMagStruct::UniquePtr event_mag_struct_msg;
  // dvs_msgs::msg::EventImage::UniquePtr event_image_msg;
  uint8_t count_R = 0;
  uint8_t reached = 0;

  RCLCPP_INFO(this->get_logger(), "Streaming Rate goal: %dHz", (int)streaming_rate_);
  if ((int)streaming_rate_ > 100)
  {
	reached = 1;
  }
  else
  {
	reached = 100 / (int)streaming_rate_;
  }

  event_mag_struct_msg = dvs_msgs::msg::EventMagStruct::UniquePtr(new dvs_msgs::msg::EventMagStruct());
  // event_image_msg = dvs_msgs::msg::EventImage::UniquePtr(new dvs_msgs::msg::EventImage());
  int pub_count=0;
  while (running_)
  {
	try
	{
	  // RCLCPP_INFO(this->get_logger(),"Getting packet");
	  caerEventPacketContainer packetContainer = caerDeviceDataGet(dvxplorer_handle_);
	  // RCLCPP_INFO(this->get_logger(),"Got packet");
	  if (packetContainer == nullptr)
	  {
		continue;  // Skip if nothing there.
	  }

	  // RCLCPP_INFO(this->get_logger(),"Getting no. packets");
	  const int32_t packetNum = caerEventPacketContainerGetEventPacketsNumber(packetContainer);

	  for (int32_t i = 0; i < packetNum; i++)
	  {
		// RCLCPP_INFO(this->get_logger(),"Getting event packet");
		caerEventPacketHeader packetHeader = caerEventPacketContainerGetEventPacket(packetContainer, i);
		if (packetHeader == nullptr)
		{
		  continue;	 // Skip if nothing there.
		}

		const int type = caerEventPacketHeaderGetEventType(packetHeader);

		// Packet 0 is always the special events packet for DVS128, while packet is the polarity events packet.
		if (type == POLARITY_EVENT)
		{
		  // if (!event_array_msg) {
		  // 	event_array_msg         = dvs_msgs::EventArrayPtr(new dvs_msgs::EventArray());
		  // 	event_array_msg->height = dvxplorer_info_.dvsSizeY;
		  // 	event_array_msg->width  = dvxplorer_info_.dvsSizeX;
		  // }

		  // RCLCPP_INFO(this->get_logger(),"make msg");
		  if (!event_mag_struct_msg)
		  {
			event_mag_struct_msg = dvs_msgs::msg::EventMagStruct::UniquePtr(new dvs_msgs::msg::EventMagStruct());

		  }

		  // if(!event_image_msg)
		  // {
		  // 	event_image_msg = dvs_msgs::msg::EventImage::UniquePtr(new dvs_msgs::msg::EventImage());
		  // }

		  // RCLCPP_INFO(this->get_logger(),"omp");
		  caerPolarityEventPacket polarity = (caerPolarityEventPacket)packetHeader;

		  const int numEvents = caerEventPacketHeaderGetEventNumber(packetHeader);

// float a_local[160000] = {0};
#pragma omp parallel num_threads(4)	 // 4 threads is the minimum number of threads to achieve 100hz
		  {
			std::vector<uint16_t> eventArr_local;
			std::vector<float> eventTime_local;
// #pragma omp for schedule(static) reduction(+:a_local)
#pragma omp for schedule(static)
			for (int j = 0; j < numEvents; j++)
			{
			  // Get full timestamp and addresses of first event.
			  caerPolarityEvent event = caerPolarityEventPacketGetEvent(polarity, j);
			  // if (j == 0) {
			  // 	event_image_msg->header.stamp = reset_time_
			  // 	+ rclcpp::Duration::from_nanoseconds(caerPolarityEventGetTimestamp64(event, polarity) * 1000);
			  // 	// + rclcpp::Duration().fromNSec(caerPolarityEventGetTimestamp64(event, polarity) * 1000);

			  // }
			  uint16_t eX = caerPolarityEventGetX(event);
			  uint16_t eY = caerPolarityEventGetY(event);
			  uint32_t key = eY * 640 + eX;

			  // If event is not one of the hot pixels, push back
			  if (!std::binary_search(hot_pixels.begin(), hot_pixels.end(), key))
			  {
				eventArr_local.push_back(eX);
				eventArr_local.push_back(eY);
				// eventArr_local.push_back(caerPolarityEventGetPolarity(event));
				eventTime_local.push_back((float)(caerPolarityEventGetTimestamp64(event, polarity) / 1e6));

				// // Load into neon registers
				// const float32x4_t v1 = vld1q_f32(a_local+key-802);
				// const float32x4_t v2 = vld1q_f32(a_local+key-402);
				// const float32x4_t v3 = vld1q_f32(a_local+key-2);
				// const float32x4_t v4 = vld1q_f32(a_local+key+398);
				// const float32x4_t v5 = vld1q_f32(a_local+key+798);

				// // vectorized code (four at once)
				// // add and return to c memory
				// float32x4_t sum = vaddq_f32(v1, gs1);
				// vst1q_f32(a_local+key-802, sum);
				// sum = vaddq_f32(v2, gs2);
				// vst1q_f32(a_local+key-402, sum);
				// sum = vaddq_f32(v3, gs3);
				// vst1q_f32(a_local+key-2, sum);
				// sum = vaddq_f32(v4, gs2);
				// vst1q_f32(a_local+key+398, sum);
				// sum = vaddq_f32(v5, gs1);
				// vst1q_f32(a_local+key+798, sum);

				// // scalar code for the remaining items.
				// a_local[key-798]+=0.002915024f;
				// a_local[key-398]+=0.013064233f;
				// a_local[key+2]+=0.021539279f;
				// a_local[key+402]+=0.013064233f;
				// a_local[key+802]+=0.002915024f;
			  }
			}

#pragma omp for schedule(static) ordered nowait
			for (int i = 0; i < omp_get_num_threads(); i++)
			{
#pragma omp ordered
			  event_mag_struct_msg->event_arr.data.insert(event_mag_struct_msg->event_arr.data.end(),
														  eventArr_local.begin(), eventArr_local.end());
			  event_mag_struct_msg->event_time.data.insert(event_mag_struct_msg->event_time.data.end(),
														   eventTime_local.begin(), eventTime_local.end());
			}

			// #pragma omp for
			// for (int j = 0; j < 160000; j++) {
			// 	event_image_msg->data[j] = a_local[j];
			// }
		  }

		  // int streaming_rate = streaming_rate_;
		  // int max_events     = max_events_;

		  // throttle event messages
		  // if ((boost::posix_time::microsec_clock::local_time() > next_send_time) || (streaming_rate == 0)
		  // 	|| ((max_events != 0) && (event_array_msg->events.size() > max_events))) {
		  // 	event_array_pub_->publish(event_array_msg);

		  // 	if (streaming_rate > 0) {
		  // 		next_send_time += delta_;
		  // 	}

		  // 	if ((max_events != 0) && (event_array_msg->events.size() > max_events)) {
		  // 		next_send_time = boost::posix_time::microsec_clock::local_time() + delta_;
		  // 	}

		  // 	event_array_msg.reset();
		  // }

		  // RCLCPP_INFO(this->get_logger(),"publish check %d/%d",count_R,reached);
		  count_R++;
		  // std_msgs::Int32 count_msg;
		  // count_msg.data = event_mag_struct_msg->eventTime.data.size();
		  // event_image_msg->size = event_mag_struct_msg->event_time.data.size();

		  // throttle event messages
		  if (count_R == reached)
		  {
			if (get_measurement_ready_drdy())
			{
			  event_mag_struct_msg->first_theta = prev_theta_;
			  struct Measurements res;
			  res = get_measurement(0, true);
			  prev_theta_ = atan2((float32_t)res.y, (float32_t)res.x) + M_PIl;
			//   RCLCPP_INFO(this->get_logger(),"%d mag count %ld %ld %f",pub_count,res.x,res.y,prev_theta_);
			  pub_count=(pub_count+1)%100;
			  event_mag_struct_msg->last_theta = prev_theta_;
			}
			// RCLCPP_INFO(this->get_logger(),"published");
			event_struct_pub_->publish(std::move(event_mag_struct_msg));
			// event_image_pub_->publish(std::move(event_image_msg));
			// event_size_pub_->publish(count_msg);
			event_mag_struct_msg.reset();
			// event_image_msg.reset();
			count_R = 0;
			// std::cout << (int)reached << std::endl;
		  }

		  // RCLCPP_INFO(this->get_logger(),"Done");
		  // if (camera_info_manager_->isCalibrated()) {
		  // 	// sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg(
		  // 	// 	new sensor_msgs::msg::CameraInfo(camera_info_manager_->getCameraInfo()));
		  // 	// camera_info_pub_->publish(camera_info_msg);
		  // 	camera_info_pub_->publish(sensor_msgs::msg::CameraInfo(camera_info_manager_->getCameraInfo()));
		  // }
		}
		// else if (type == IMU6_EVENT) {
		// 	caerIMU6EventPacket imu = (caerIMU6EventPacket) packetHeader;

		// 	const int numEvents = caerEventPacketHeaderGetEventNumber(packetHeader);

		// 	for (int j = 0; j < numEvents; j++) {
		// 		caerIMU6Event event = caerIMU6EventPacketGetEvent(imu, j);

		// 		sensor_msgs::msg::Imu msg;

		// 		// convert from g's to m/s^2 and align axes with camera frame
		// 		msg.linear_acceleration.x = -caerIMU6EventGetAccelX(event) * STANDARD_GRAVITY;
		// 		msg.linear_acceleration.y = caerIMU6EventGetAccelY(event) * STANDARD_GRAVITY;
		// 		msg.linear_acceleration.z = -caerIMU6EventGetAccelZ(event) * STANDARD_GRAVITY;
		// 		// convert from deg/s to rad/s and align axes with camera frame
		// 		msg.angular_velocity.x = -caerIMU6EventGetGyroX(event) / 180.0 * M_PI;
		// 		msg.angular_velocity.y = caerIMU6EventGetGyroY(event) / 180.0 * M_PI;
		// 		msg.angular_velocity.z = -caerIMU6EventGetGyroZ(event) / 180.0 * M_PI;

		// 		// no orientation estimate: http://docs.ros.org/api/sensor_msgs/html/msg/Imu.html
		// 		msg.orientation_covariance[0] = -1.0;

		// 		// time
		// 		msg.header.stamp
		// 			= reset_time_ + rclcpp::Duration::from_nanoseconds(caerIMU6EventGetTimestamp64(event, imu) * 1000);

		// 		// frame
		// 		msg.header.frame_id = "base_link";

		// 		// IMU calibration
		// 		if (imu_calibration_running_) {
		// 			if ((int)imu_calibration_samples_.size() < imu_calibration_sample_size_) {
		// 				imu_calibration_samples_.push_back(msg);
		// 			}
		// 			else {
		// 				imu_calibration_running_ = false;
		// 				updateImuBias();
		// 			}
		// 		}

		// 		// bias correction
		// 		msg.linear_acceleration.x -= bias.linear_acceleration.x;
		// 		msg.linear_acceleration.y -= bias.linear_acceleration.y;
		// 		msg.linear_acceleration.z -= bias.linear_acceleration.z;
		// 		msg.angular_velocity.x -= bias.angular_velocity.x;
		// 		msg.angular_velocity.y -= bias.angular_velocity.y;
		// 		msg.angular_velocity.z -= bias.angular_velocity.z;

		// 		imu_pub_->publish(msg);
		// 	}
		// }
	  }

	  caerEventPacketContainerFree(packetContainer);

	  // rclcpp::spin_some(this->get_node_base_interface());
	}
	catch (boost::thread_interrupted&)
	{
	  return;
	}
  }

  caerDeviceDataStop(dvxplorer_handle_);
}

}  // namespace dvxplorer_ros_driver