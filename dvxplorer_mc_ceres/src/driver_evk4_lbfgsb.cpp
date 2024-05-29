// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_mc_ceres/driver_evk4_lbfgsb.h"

#include <std_msgs/Int32.h>
namespace dvxplorer_mc_ceres
{

	void MyProcessor::eventCD(uint64_t t, uint16_t ex, uint16_t ey, uint8_t)
	{
		if (frame_processed_)
		{
			if (t < t_new_frame_)
			{
				return;
			}
			else{
				frame_processed_=false;
				reached_first_midpoint_frame_=false;
				reached_last_midpoint_frame_=false;
				last_event_idx_==-1;
			}
		}
		else if (reached_first_midpoint_frame_&&!reached_last_midpoint_frame_)
		{
			if(t>t_midpoint_){
				reached_last_midpoint_frame_=true;
				last_event_idx_=(mc_gr->num_events_+first_midpoint_event_idx_)/2 + mc_gr->max_num_events_/2;
			}
		}
		else if (t_new_frame_ == -1)
		{
			t_new_frame_ = t;
			t_midpoint_=t_new_frame_+0.5e-2;
		}
		else if(!reached_first_midpoint_frame_&&t>=t_midpoint_){
			reached_first_midpoint_frame_=true;
			first_midpoint_event_idx_=mc_gr_->num_events_;
		}


		// TODO: equal events before and after middle ts
		//  mc_gr_->AddData(t,ex-440,ey-160);
		mc_gr_->AddData(t, ex + x_offset_, ey + y_offset_);
		// *outfile_<<t<<","<<ex<<","<<ey<<","<<0<<std::endl;
		// if (mc_gr_->ReadyToMC())
		if (mc_gr_->num_events_==last_event_idx_)
		{
			idx++;

			// Initial guess
			float fx;
			try
			{
				solver_.minimize(*mc_gr_, rotations_, fx, lb_, ub_);
			}
			catch (std::exception &e)
			{
				std::string error_str = e.what();
				if (error_str == "the line search step became smaller than the minimum value allowed")
				{
					ROS_INFO("<min");
				}
				else
				{
					ROS_INFO("%s", error_str.c_str());
				}
			}
			ROS_INFO("%d iterations", mc_gr_->iterations);

			ROS_INFO("Rotations: %f %f %f", rotations_[0], rotations_[1], rotations_[2]);
			uint8_t image[height_ * width_];
			float contrast;
			sensor_msgs::CompressedImage compressed;
			compressed.format = "jpeg";
			mc_gr_->GenerateImage(rotations_.data(), image, contrast);
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
			t_new_frame_ += 1e-2;
			t_midpoint_ += 1e-2;
			frame_processed_=true;
		}
	}
	MyProcessor::MyProcessor(ros::NodeHandle &nh) : tjhandle_(makeTjhandleUniquePtr()), nh_(nh), solver_(LBFGSBSolver<float>(param_))
	{
		// google::InitGoogleLogging("mc_ceres");
		mc_gr_ = std::make_shared<McGradient>(fx_, fy_, cx_, cy_, height_, width_);
		mc_gr_->allocate();

		param_.m = 10;
		// param_.epsilon = 1e-2;
		// param.epsilon_rel = 1e-3;
		param_.max_iterations = 20;
		param_.delta = 1e-4;

		rotations_[1] = -20;

		// c1 aka sufficient decrease
		// param_.ftol = 1e-6;

		// c2 aka curvature
		// param_.wolfe = 0.9;
		param_.min_step = 1e-5;
		param_.max_step = 1e3;

		ns = ros::this_node::getNamespace();
		if (ns == "/")
		{
			ns = "/dvs";
		}

		compensated_image_pub_ = nh_.advertise<sensor_msgs::CompressedImage>(ns + "/comprensated/image/compressed", 10);
	}

	MyProcessor::~MyProcessor()
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
		processor_ = std::make_shared<MyProcessor>(nh_);
	}
	DvxplorerMcCeres::~DvxplorerMcCeres()
	{
	}

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