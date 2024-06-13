// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_mc_ceres/driver_evk4.h"

#include <std_msgs/Int32.h>
namespace dvxplorer_mc_ceres
{

	void EventProcessor::eventCD(uint64_t t, uint16_t ex, uint16_t ey, uint8_t)
	{
		// mc_gr_->AddData(t,ex-440,ey-160);
		mc_gr_->AddData(t, ex + x_offset_, ey + y_offset_);
		// *outfile_<<t<<","<<ex<<","<<ey<<","<<0<<std::endl;
		if (mc_gr_->ReadyToMC())
		{
			idx++;
			// outfile_->close();
			// std::stringstream run_name;
			// run_name << "bag_" << std::setfill('0') << std::setw(5) << idx << ".csv";
			// outfile_ = std::make_shared<std::ofstream>(run_name.str(), std::ios::out);

			// rotations_[0] = 1;
			// rotations_[1]=-5*2*M_PI;
			// rotations_[1] = 1;
			// rotations_[2] = 1;
			ceres::Solve(options_, *problem_, rotations_, &summary_);
			// ROS_INFO(summary_.FullReport().c_str());
			// ROS_INFO("solution usable? %d",summary_.IsSolutionUsable());
			// ROS_INFO("%s",summary_.message.c_str());
			ROS_INFO("%d iterations", summary_.num_gradient_evaluations);
			ROS_INFO("%s",summary_.message.c_str());

			ROS_INFO("Rotations: %f %f %f", rotations_[0], rotations_[1], rotations_[2]);
			// if (!summary_.IsSolutionUsable() || abs(rotations_[0]) > 50 || abs(rotations_[1]) > 50 || abs(rotations_[2]) > 50)
			// {
			// 	std::fill_n(rotations_, 3, 0.001);
			// }
			uint8_t image[height_ * width_];
			float contrast;

			// sensor_msgs::CompressedImagePtr compressed_image_msg=sensor_msgs::CompressedImagePtr(new sensor_msgs::CompressedImage());
			sensor_msgs::CompressedImage compressed;
			compressed.format = "jpeg";
			// compressed_image_msg->format="jpeg";
			mc_gr_->GenerateImage(rotations_, image, contrast);
			unsigned char *jpeg_buffer = nullptr;
			uint64_t jpeg_size{};
			std::string targetFormat = "jpeg";
			tjCompress2(
				*tjhandle_,
				image,
				width_,
				width_,
				height_,
				TJPF_GRAY,
				&jpeg_buffer,
				&jpeg_size,
				TJSAMP_GRAY,
				100,
				TJFLAG_FASTDCT);
			// compressed_image_msg->data.resize(jpeg_size);
			compressed.data.resize(jpeg_size);
			std::copy(jpeg_buffer, jpeg_buffer + jpeg_size, compressed.data.begin());
			tjFree(jpeg_buffer);
			compensated_image_pub_.publish(compressed);
			mc_gr_->ClearEvents();
		}
	}
	EventProcessor::EventProcessor(ros::NodeHandle &nh) : tjhandle_(makeTjhandleUniquePtr()), nh_(nh)
	{
		google::InitGoogleLogging("mc_ceres");
		mc_gr_ = std::make_shared<McGradient>(fx_, fy_, cx_, cy_, height_, width_);
		mc_gr_->allocate();
		// STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT, BFGS and LBFGS.
		options_.line_search_direction_type = line_search_direction_type_;
		//  ARMIJO and WOLFE
		options_.line_search_type = line_search_type_;
		// FLETCHER_REEVES, POLAK_RIBIERE and HESTENES_STIEFEL
		options_.nonlinear_conjugate_gradient_type = nonlinear_conjugate_gradient_type_;
		options_.max_num_line_search_step_size_iterations = 20;
		options_.function_tolerance = 1e-4;
		options_.parameter_tolerance = 1e-6;
		options_.min_line_search_step_contraction = 0.9;
		options_.max_line_search_step_contraction = 1e-3;
		options_.minimizer_progress_to_stdout = false;

		// options_.minimizer_progress_to_stdout = true;
		problem_ = std::make_shared<ceres::GradientProblem>(mc_gr_.get());

		ns = ros::this_node::getNamespace();
		if (ns == "/")
		{
			ns = "/dvs";
		}

		compensated_image_pub_ = nh_.advertise<sensor_msgs::CompressedImage>(ns + "/comprensated/image/compressed", 10);

		// std::stringstream run_name;
		// run_name << "bag_" << std::setfill('0') << std::setw(5) << idx << ".csv";
		// std::cout << run_name.str() << std::endl;
		// outfile_ = std::make_shared<std::ofstream>(run_name.str(), std::ios::out);
	}

	EventProcessor::~EventProcessor()
	{
	}
	DvxplorerMcCeres::DvxplorerMcCeres(ros::NodeHandle &nh, ros::NodeHandle nh_private) : nh_(nh)
	{
		// load parameters
		double reset_timestamps_delay;

		// set namespace
		ns = ros::this_node::getNamespace();
		if (ns == "/")
		{
			ns = "/dvs";
		}

		// event_struct_sub_ = nh_.subscribe<dvs_msgs::EventStruct>(ns+ "/eventStruct",10,&DvxplorerMcCeres::eventStructCallback,this);
		event_struct_sub_ = nh_.subscribe("/event_camera/events", 10, &DvxplorerMcCeres::eventStructCallback, this);
		processor_ = std::make_shared<EventProcessor>(nh_);
	}
	DvxplorerMcCeres::~DvxplorerMcCeres()
	{
	}

	// void DvxplorerMcCeres::eventStructCallback(const dvs_msgs::EventStruct::ConstPtr &msg)
	// {
	// 	mc_gr_->reset();
	// 	mc_gr_->ReplaceData(msg);

	// 	ceres::Solve(options_, *problem_, rotations_, &summary_);
	// 	// ROS_INFO(summary_.FullReport().c_str());
	// 	// ROS_INFO("solution usable? %d",summary_.IsSolutionUsable());
	// 	ROS_INFO("%s",summary_.message.c_str());
	// 	ROS_INFO("%d iterations", summary_.num_gradient_evaluations);

	// 	ROS_INFO("Rotations: %f %f %f", rotations_[0], rotations_[1], rotations_[2]);
	// 	std::cout << "rot : " << rotations_[0] << " " << rotations_[1] << " " << rotations_[2] << " "
	// 			  << "\n";
	// 	if (!summary_.IsSolutionUsable() || abs(rotations_[0]) > 10 || abs(rotations_[1]) > 10 || abs(rotations_[2]) > 10)
	// 	{
	// 		std::fill_n(rotations_, 3, 0.001);
	// 	}
	// 	uint8_t image[height_ * width_];
	// 	float contrast;

	// 	// sensor_msgs::CompressedImagePtr compressed_image_msg=sensor_msgs::CompressedImagePtr(new sensor_msgs::CompressedImage());
	// 	sensor_msgs::CompressedImage compressed;
	// 	compressed.format = "jpeg";
	// 	// compressed_image_msg->format="jpeg";
	// 	mc_gr_->GenerateImage(rotations_, image, contrast);
	// 	unsigned char *jpeg_buffer = nullptr;
	// 	uint64_t jpeg_size{};
	// 	std::string targetFormat = "jpeg";
	// 	tjCompress2(
	// 		*tjhandle_,
	// 		image,
	// 		width_,
	// 		width_,
	// 		height_,
	// 		TJPF_GRAY,
	// 		&jpeg_buffer,
	// 		&jpeg_size,
	// 		TJSAMP_GRAY,
	// 		100,
	// 		TJFLAG_FASTDCT);
	// 	// compressed_image_msg->data.resize(jpeg_size);
	// 	compressed.data.resize(jpeg_size);
	// 	std::copy(jpeg_buffer, jpeg_buffer + jpeg_size, compressed.data.begin());
	// 	tjFree(jpeg_buffer);
	// 	compensated_image_pub_.publish(compressed);
	// }
	void DvxplorerMcCeres::eventStructCallback(const event_camera_codecs::EventPacketConstSharedPtr &msg)
	{
		// will create a new decoder on first call, from then on returns existing one
		auto decoder = decoderFactory_.getInstance(*msg);
		if (!decoder)
		{ // msg->encoding was invalid
			return;
		}
		// the decode() will trigger callbacks to processor
		decoder->decode(*msg, processor_.get());
		// ROS_INFO("got a message");
	}
} // namespace dvxplorer_mc_ceres