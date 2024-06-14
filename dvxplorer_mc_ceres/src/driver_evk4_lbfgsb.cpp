// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_mc_ceres/driver_evk4_lbfgsb.h"

#include <std_msgs/Int32.h>
namespace dvxplorer_mc_ceres
{

	// void EventProcessor::eventCD(uint64_t t, uint16_t ex, uint16_t ey, uint8_t)
	// {
	// 	events_++;
	// 	if (prev_t_ == 0)
	// 	{
	// 		prev_t_ = t;
	// 		whole_begin_ = ros::Time::now();
	// 	}
	// 	else if (t > prev_t_ + 10 * 1e6)
	// 	{
	// 		ros::Time decode_end = ros::Time::now();
	// 		double decode_duration = (decode_end - whole_begin_).toSec();
	// 		prev_t_ += 10 * 1e6;
	// 		ROS_INFO("%d", events_);
	// 		events_ = 0;

	// 		ROS_INFO("decoding took %lf seconds", decode_duration);
	// 		whole_begin_ = ros::Time::now();
	// 	}
	// }
	void EventProcessor::optimizerLoop()
	{
		while (ros::ok())
		{
			std::unique_lock<std::mutex> lk(mutex_);
			cv_.wait(lk);
			ROS_INFO("optimizer woke");
			if(ros::isShuttingDown()){
				break;
			}
			ros::Time waiting_end = ros::Time::now();

			idx++;
			ros::Time solver_begin = ros::Time::now();
			// Initial guess
			float fx;
			try
			{
				solver_.minimize(* mc_gr_[ready_mc_gr_idx_], rotations_, fx, lb_, ub_, 9000);
			}
			catch (std::exception &e)
			{
				std::string error_str = e.what();
				if (error_str == "the line search step became smaller than the minimum value allowed")
				{
					// ROS_INFO("<min");
				}
				else
				{
					ROS_INFO("%s", error_str.c_str());
				}
			}
			ros::Time solver_end = ros::Time::now();
			// ROS_INFO("%d iterations", mc_gr_->iterations);

			// ROS_INFO("Rotations: %f %f %f, t=%fms", rotations_[0], rotations_[1], rotations_[2],mc_gr_->approx_middle_t_/1e6);


			// uint8_t image[height_ * width_];
			float contrast;
			sensor_msgs::CompressedImage compressed;
			compressed.format = "jpeg";
			mc_gr_[ready_mc_gr_idx_]->GenerateImage(rotations_.data(), contrast);
			unsigned char *jpeg_buffer = nullptr;
			uint64_t jpeg_size{};
			std::string targetFormat = "jpeg";
			tjCompress2(
				*tjhandle_,
				mc_gr_[ready_mc_gr_idx_]->output_image,
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
			geometry_msgs::TwistStamped msg;
			msg.twist.angular.x = rotations_[0];
			msg.twist.angular.y = rotations_[1];
			msg.twist.angular.z = rotations_[2];
			msg.header.stamp.sec = mc_gr_[ready_mc_gr_idx_]->GetMiddleT() / 1e9;
			msg.header.stamp.nsec = mc_gr_[ready_mc_gr_idx_]->GetMiddleT() - msg.header.stamp.sec * 1e9;
			velocity_pub_.publish(msg);
			mc_gr_[ready_mc_gr_idx_]->ClearEvents();

			ros::Time whole_end = ros::Time::now();
			double postprocessing_duration = (whole_end - solver_end).toSec();
			double whole_duration = (whole_end - whole_begin_).toSec();
			double waiting_duration = (waiting_end - whole_begin_).toSec();
			double solver_duration = (solver_end - solver_begin).toSec();
			ROS_INFO("waited for %lf seconds", waiting_duration);
			ROS_INFO("optimization took %lf seconds", solver_duration);
			ROS_INFO("postprocessing took %lf seconds", postprocessing_duration);
			ROS_INFO("entire callback took %lf seconds", whole_duration);
			whole_begin_ = ros::Time::now();
			lk.unlock();
		}
	}
	void EventProcessor::eventCD(uint64_t t, uint16_t ex, uint16_t ey, uint8_t)
	{
		if (mc_gr_[in_progress_mc_gr_idx_]->ReadyToMC(t))
		{
			// ROS_INFO("ready to mc");
			if (mc_gr_[in_progress_mc_gr_idx_]->SufficientEvents())
			{
				mc_gr_[in_progress_mc_gr_idx_]->syncClearImages();
				// ROS_INFO("sufficient events");
				{
					std::lock_guard<std::mutex> lk(mutex_);
					// ROS_INFO("lock obtained");
					ready_mc_gr_idx_=in_progress_mc_gr_idx_;
					in_progress_mc_gr_idx_=(in_progress_mc_gr_idx_+1)%max_mc_gr_idx_;
				}
				cv_.notify_one();
				mc_gr_[in_progress_mc_gr_idx_]->clearImages();
			}
			else
			{
				ROS_INFO("skipped solver, insufficient events");
				mc_gr_[in_progress_mc_gr_idx_]->ClearEvents();
			}
		}
		mc_gr_[in_progress_mc_gr_idx_]->AddData(t, ex, ey);
		
		if (new_config_)
		{
			if (new_height_ != height_ || new_width_ != width_)
			{
				height_ = new_height_;
				width_ = new_width_;
				int new_x_offset = (1280 - new_width_) / 2;
				int new_y_offset = (720 - new_height_) / 2;
				mc_gr_[in_progress_mc_gr_idx_]->reset(height_, width_, new_x_offset, new_y_offset);
			}
			new_config_ = false;
		}
	}
	EventProcessor::EventProcessor(ros::NodeHandle &nh) : tjhandle_(makeTjhandleUniquePtr()), nh_(nh), solver_(LBFGSBSolver<float>(param_))
	{
		// google::InitGoogleLogging("mc_ceres");
		for(int i=0; i<2; i++){
			mc_gr_.push_back(std::make_shared<McGradient>(fx_, fy_, cx_, cy_, height_, width_));
			mc_gr_.back()->allocate();
		}
		// mc_gr_ = std::make_shared<McGradient>(fx_, fy_, cx_, cy_, height_, width_);
		// mc_gr_->allocate();

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
		velocity_pub_ = nh_.advertise<geometry_msgs::TwistStamped>(ns + "/velocity", 10);
		whole_begin_ = ros::Time::now();
		optimizer_thread_=std::make_shared<std::thread>(&EventProcessor::optimizerLoop, this);
	}
	void EventProcessor::reconfigure(dvxplorer_mc_ceres::DVXplorer_MC_CeresConfig &config)
	{
		new_height_ = config.height;
		new_width_ = config.width;
		// new_x_offset_ = config.x_offset;
		// new_y_offset_ = config.y_offset;
		new_config_ = true;
	}
	EventProcessor::~EventProcessor()
	{
		cv_.notify_all();
  		optimizer_thread_->join();
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
		event_struct_sub_ = nh_.subscribe("/event_camera/events", 1000, &DvxplorerMcCeres::eventStructCallback, this);
		processor_ = std::make_shared<EventProcessor>(nh_);

		// Dynamic reconfigure
		// Will call callback, which will pass stored config to device.
		dynamic_reconfigure_callback_ = boost::bind(&DvxplorerMcCeres::callback, this, _1, _2);
		server_.reset(new dynamic_reconfigure::Server<dvxplorer_mc_ceres::DVXplorer_MC_CeresConfig>(nh_private));
		server_->setCallback(dynamic_reconfigure_callback_);
	}
	DvxplorerMcCeres::~DvxplorerMcCeres()
	{
	}

	void DvxplorerMcCeres::callback(dvxplorer_mc_ceres::DVXplorer_MC_CeresConfig &config, uint32_t level)
	{
		processor_->reconfigure(config);
		// config.height;
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