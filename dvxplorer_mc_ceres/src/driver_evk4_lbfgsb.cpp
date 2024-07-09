// This file is part of DVS-ROS - the RPG DVS ROS Package

#include "dvxplorer_mc_ceres/driver_evk4_lbfgsb.h"

#include <std_msgs/Int32.h>
namespace dvxplorer_mc_ceres
{

	void EventProcessor::optimizerLoop()
	{
		while (ros::ok())
		{
			std::unique_lock<std::mutex> lk(mutex_);
			cv_.wait(lk);
			// ROS_INFO("optimizer woke");
			if (ros::isShuttingDown())
			{
				break;
			}
			ros::Time waiting_end = ros::Time::now();

			idx++;
			ros::Time solver_begin = ros::Time::now();
			// Initial guess
			float fx;
			int gaussian_iters, bilinear_iters;
			try
			{
				mc_gr_[ready_mc_gr_idx_]->iterations = 0;
				solver_.minimize(*mc_gr_[ready_mc_gr_idx_], rotations_, fx, lb_, ub_, 9000);
				gaussian_iters = mc_gr_[ready_mc_gr_idx_]->iterations;
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
					// ROS_INFO("%s", error_str.c_str());
				}
			}

			ros::Time solver_end_gaussian = ros::Time::now();
			try
			{
				mc_gr_[ready_mc_gr_idx_]->iterations = 0;
				mc_gr_[ready_mc_gr_idx_]->setUseBilinear(true);
				solver_.minimize(*mc_gr_[ready_mc_gr_idx_], rotations_bilinear_, fx, lb_, ub_, 9000);
				bilinear_iters = mc_gr_[ready_mc_gr_idx_]->iterations;
				mc_gr_[ready_mc_gr_idx_]->setUseBilinear(false);
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
					// ROS_INFO("%s", error_str.c_str());
				}
			}
			ros::Time solver_end = ros::Time::now();
			// ROS_INFO("Rotations gaussian 5x5 : %f %f %f, t=%fms, %d iterations", rotations_[0], rotations_[1], rotations_[2],mc_gr_[ready_mc_gr_idx_]->GetApproxMiddleTs()/1e6, gaussian_iters);
			// ROS_INFO("Rotations bilinear     : %f %f %f, t=%fms, %d iterations", rotations_bilinear_[0], rotations_bilinear_[1], rotations_bilinear_[2],mc_gr_[ready_mc_gr_idx_]->GetApproxMiddleTs()/1e6, bilinear_iters);

			// uint8_t image[height_ * width_];
			float contrast;
			sensor_msgs::CompressedImage compressed;
			compressed.format = "jpeg";
            Eigen::VectorXd rotations_double=rotations_.cast<double>();
			mc_gr_[ready_mc_gr_idx_]->GenerateImage(rotations_double.data());

			// mc_gr_[ready_mc_gr_idx_]->GenerateUncompensatedImage(contrast);
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
			geometry_msgs::TwistStamped msg;
			compensated_image_pub_.publish(compressed);
			msg.twist.angular.x = rotations_[0];
			msg.twist.angular.y = rotations_[1];
			msg.twist.angular.z = rotations_[2];
			msg.header.stamp.sec = mc_gr_[ready_mc_gr_idx_]->GetApproxMiddleT() / 1e9;
			msg.header.stamp.nsec = mc_gr_[ready_mc_gr_idx_]->GetApproxMiddleT() - msg.header.stamp.sec * 1e9;
			velocity_pub_.publish(msg);

			msg.twist.angular.x = rotations_bilinear_[0];
			msg.twist.angular.y = rotations_bilinear_[1];
			msg.twist.angular.z = rotations_bilinear_[2];
			velocity_pub_bilinear_.publish(msg);

			mc_gr_[ready_mc_gr_idx_]->ClearEvents();

			ros::Time whole_end = ros::Time::now();
			double postprocessing_duration = (whole_end - solver_end).toSec();
			double whole_duration = (whole_end - whole_begin_).toSec();
			double waiting_duration = (waiting_end - whole_begin_).toSec();
			double solver_gaussian_duration = (solver_end_gaussian - solver_begin).toSec();
			double solver_bilinear_duration = (solver_end - solver_end_gaussian).toSec();
			// ROS_INFO("waited for %lf seconds", waiting_duration);
			// ROS_INFO("optimization gaussian took %lf seconds", solver_gaussian_duration);
			// ROS_INFO("optimization bilinear took %lf seconds", solver_bilinear_duration);
			// ROS_INFO("postprocessing took %lf seconds", postprocessing_duration);
			// ROS_INFO("entire callback took %lf seconds", whole_duration);
			whole_begin_ = ros::Time::now();
			lk.unlock();
		}
	}
	void EventProcessor::eventCD(uint64_t t, uint16_t ex, uint16_t ey, uint8_t)
	{
		total_events_++;
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
					ready_mc_gr_idx_ = in_progress_mc_gr_idx_;
					in_progress_mc_gr_idx_ = (in_progress_mc_gr_idx_ + 1) % max_mc_gr_idx_;
				}
				cv_.notify_one();
				
				if (mc_gr_[in_progress_mc_gr_idx_]->reconfigure_pending)
				{
					mc_gr_[in_progress_mc_gr_idx_]->reset(height_, width_, x_offset_, y_offset_);
				}
				mc_gr_[in_progress_mc_gr_idx_]->clearImages();
				mc_gr_[in_progress_mc_gr_idx_]->SetTimestampGoals(mc_gr_[ready_mc_gr_idx_]->GetApproxMiddleTs() + (int64_t)10000000);
			}
			else
			{
				// ROS_INFO("skipped solver for ts %ld, insufficient events %d",mc_gr_[in_progress_mc_gr_idx_]->GetApproxMiddleTs(),mc_gr_[in_progress_mc_gr_idx_]->getNumEventsToBeConsidered);
				mc_gr_[in_progress_mc_gr_idx_]->ClearEvents();
				mc_gr_[in_progress_mc_gr_idx_]->SetTimestampGoals(mc_gr_[in_progress_mc_gr_idx_]->GetApproxMiddleTs() + (int64_t)10000000);
			}
		}
		mc_gr_[in_progress_mc_gr_idx_]->AddData(t, ex, ey);

	}
	EventProcessor::EventProcessor(ros::NodeHandle &nh) : tjhandle_(makeTjhandleUniquePtr()), nh_(nh), solver_(LBFGSBSolver<float>(param_))
	{
		// google::InitGoogleLogging("mc_ceres");
		for (int i = 0; i < 2; i++)
		{
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
		rotations_bilinear_[1] = -20;

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
		velocity_pub_bilinear_ = nh_.advertise<geometry_msgs::TwistStamped>(ns + "/velocity_bilinear", 10);
		whole_begin_ = ros::Time::now();
		optimizer_thread_ = std::make_shared<std::thread>(&EventProcessor::optimizerLoop, this);
	}
	void EventProcessor::reconfigure(dvxplorer_mc_ceres::DVXplorer_MC_CeresConfig &config)
	{
		if(config.height!=height_||config.width!=width_){
			height_ = config.height;
			width_ = config.width;
			
			x_offset_ = (1280 - new_width_) / 2;
			y_offset_ = (720 - new_height_) / 2;
			for(int i=0; i<max_mc_gr_idx_;i++){
				mc_gr_[i]->reconfigure_pending=true;
			}
		}
	}
	EventProcessor::~EventProcessor()
	{
		cv_.notify_all();
		optimizer_thread_->join();
	}

	uint64_t EventProcessor::getTotalEvents() { return total_events_; }

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

		// ROS_INFO("total events processed: %ld", processor_->getTotalEvents());
	}

} // namespace dvxplorer_mc_ceres