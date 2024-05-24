// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_mc_ceres/driver.h"

#include <std_msgs/Int32.h>

namespace dvxplorer_mc_ceres
{

	DvxplorerMcCeres::DvxplorerMcCeres(ros::NodeHandle &nh, ros::NodeHandle nh_private) : nh_(nh), tjhandle_(makeTjhandleUniquePtr())
	{
		google::InitGoogleLogging("mc_ceres");
		// load parameters
		double reset_timestamps_delay;

		// set namespace
		ns = ros::this_node::getNamespace();
		if (ns == "/")
		{
			ns = "/dvs";
		}

		// event_struct_sub_ = nh_.subscribe<dvs_msgs::EventStruct>(ns+ "/eventStruct",10,&DvxplorerMcCeres::eventStructCallback,this);
		event_struct_sub_ = nh_.subscribe(ns + "/eventStruct", 10, &DvxplorerMcCeres::eventStructCallback, this);
		compensated_image_pub_ = nh_.advertise<sensor_msgs::CompressedImage>(ns + "/comprensated/image/compressed", 10);
		mc_gr_ = std::make_shared<McGradient>(fx_, fy_, cx_, cy_, height_, width_);

		// STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT, BFGS and LBFGS.
		options_.line_search_direction_type = line_search_direction_type_;
		//  ARMIJO and WOLFE
		options_.line_search_type = line_search_type_;
		// FLETCHER_REEVES, POLAK_RIBIERE and HESTENES_STIEFEL
		options_.nonlinear_conjugate_gradient_type = nonlinear_conjugate_gradient_type_;
		options_.max_num_line_search_step_size_iterations = 20;
		options_.function_tolerance = 1e-5;
		options_.parameter_tolerance = 1e-6;
		options_.minimizer_progress_to_stdout = false;

		// options_.minimizer_progress_to_stdout = true;
		problem_ = std::make_shared<ceres::GradientProblem>(mc_gr_.get());
	}
	DvxplorerMcCeres::~DvxplorerMcCeres()
	{
	}

	void DvxplorerMcCeres::eventStructCallback(const dvs_msgs::EventStruct::ConstPtr &msg)
	{
		// int num_events = msg->eventTime.data.size();
		// std::stringstream outfile_name;
		// outfile_name<<"out"<<packet_count_<<".csv";
    	// std::ofstream outfile(outfile_name.str(), std::ios::out);
		// packet_count_++;
		// for (int i = 0; i < num_events; i++)
        // {
        //     outfile<<msg->eventTime.data[i]<<","<<msg->eventArr.data[i*2]<<","<<msg->eventArr.data[i*2+1]<<",0"<<std::endl;
        // }
		// outfile.close();
		// if(packet_count_>10){
		// 	ros::shutdown();
		// }
		// return;
		// std::cout<<msg->eventArr.layout.dim[0].size<<std::endl;
		// std::cout << msg->eventTime.data.size() << std::endl;
		mc_gr_->reset();
		mc_gr_->ReplaceData(msg);

		// std::vector<ceres::LineSearchDirectionType> line_search_direction_types{ceres::STEEPEST_DESCENT, ceres::NONLINEAR_CONJUGATE_GRADIENT, ceres::LBFGS, ceres::BFGS};
		// std::vector<ceres::LineSearchType> line_search_types{ceres::ARMIJO, ceres::WOLFE};
		// std::vector<ceres::NonlinearConjugateGradientType> nonlinear_conjugate_gradient_types{ceres::FLETCHER_REEVES, ceres::POLAK_RIBIERE, ceres::HESTENES_STIEFEL};
		// for (auto line_search_direction_type : line_search_direction_types)
		// {
		// 	for (auto line_search_type : line_search_types)
		// 	{
		// 		for (auto nonlinear_conjugate_gradient_type : nonlinear_conjugate_gradient_types)
		// 		{
		// 			for (int use_approximate_eigenvalue_bfgs_scaling = 0; use_approximate_eigenvalue_bfgs_scaling < 2; use_approximate_eigenvalue_bfgs_scaling++)
		// 			{
		// 				for (int interpolation = 0; interpolation < 3; interpolation++)
		// 				{

		// 					options_.use_approximate_eigenvalue_bfgs_scaling = use_approximate_eigenvalue_bfgs_scaling;
		// 					// STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT, BFGS and LBFGS.
		// 					options_.line_search_direction_type = line_search_direction_type;
		// 					//  ARMIJO and WOLFE
		// 					options_.line_search_type = line_search_type;
		// 					// FLETCHER_REEVES, POLAK_RIBIERE and HESTENES_STIEFEL
		// 					options_.nonlinear_conjugate_gradient_type = nonlinear_conjugate_gradient_type;
		// 					std::string s;
		// 					if (!options_.IsValid(&s))
		// 					{
		// 						continue;
		// 					}
		// 					ROS_INFO("using %d %d %d %d %d", line_search_direction_type, line_search_type, nonlinear_conjugate_gradient_type, use_approximate_eigenvalue_bfgs_scaling,interpolation);
		// 					std::fill_n(rotations_, 3, 0.001);
		// 					ceres::Solve(options_, *problem_, rotations_, &summary_);
		// 					// ROS_INFO(summary_.FullReport().c_str());
		// 					// ROS_INFO("solution usable? %d",summary_.IsSolutionUsable());
		// 					ROS_INFO(summary_.message.c_str());
		// 					ROS_INFO("%d iterations", summary_.num_gradient_evaluations);

		// 					ROS_INFO("Rotations: %f %f %f", rotations_[0], rotations_[1], rotations_[2]);
		// 				}
		// 			}
		// 		}
		// 	}
		// }
		ceres::Solve(options_, *problem_, rotations_, &summary_);
		// ROS_INFO(summary_.FullReport().c_str());
		// ROS_INFO("solution usable? %d",summary_.IsSolutionUsable());
		ROS_INFO(summary_.message.c_str());
		ROS_INFO("%d iterations", summary_.num_gradient_evaluations);

		ROS_INFO("Rotations: %f %f %f", rotations_[0], rotations_[1], rotations_[2]);
		// std::cout << "rot : " << rotations_[0] << " " << rotations_[1] << " " << rotations_[2] << " "
		// 		  << "\n";
		if (!summary_.IsSolutionUsable() || abs(rotations_[0]) > 10 || abs(rotations_[1]) > 10 || abs(rotations_[2]) > 10)
		{
			std::fill_n(rotations_, 3, 0.001);
		}
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
	}

} // namespace dvxplorer_mc_ceres