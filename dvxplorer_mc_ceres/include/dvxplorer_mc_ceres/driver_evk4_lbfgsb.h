// This file is part of DVS-ROS - the RPG DVS ROS Package

#pragma once

#include <ros/ros.h>
#include <string>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
// messages
#include <dvs_msgs/EventStruct.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/CompressedImage.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Time.h>
#include <geometry_msgs/TwistStamped.h>
#include <turbojpeg.h>

#include <event_camera_codecs/decoder.h>
#include <event_camera_codecs/evt3_decoder.h>
#include <event_camera_codecs/decoder_factory.h>

// dynamic reconfigure
#include <dvxplorer_mc_ceres/DVXplorer_MC_CeresConfig.h>
#include <dynamic_reconfigure/server.h>
// camera info manager
#include <camera_info_manager/camera_info_manager.h>
#include <sensor_msgs/CameraInfo.h>

#include "dvxplorer_mc_ceres/mc_gradient_lbfgsb.h"

namespace dvxplorer_mc_ceres
{

  using namespace LBFGSpp;
  using Eigen::VectorXf;
  struct TjhandleDeleter
  {
    void operator()(tjhandle *handle)
    {
      if (handle)
      {
        tjDestroy(*handle);
        delete handle;
      }
    }
  };

  using TjhandleUniquePtr = std::unique_ptr<tjhandle, TjhandleDeleter>;
  inline TjhandleUniquePtr makeTjhandleUniquePtr()
  {
    TjhandleUniquePtr ptr(new tjhandle, TjhandleDeleter());
    *ptr = tjInitCompress();
    return ptr;
  }

  using event_camera_codecs::EventPacket;
  class EventProcessor : public event_camera_codecs::EventProcessor
  {
  public:
    EventProcessor(ros::NodeHandle &nh);
    ~EventProcessor();
    void eventCD(uint64_t t, uint16_t ex, uint16_t ey, uint8_t) override;
    void eventExtTrigger(uint64_t, uint8_t, uint8_t) override {}
    void finished() override{};                    // called after no more events decoded in this packet
    void rawData(const char *, size_t) override{}; // passthrough of raw data
    void reconfigure(dvxplorer_mc_ceres::DVXplorer_MC_CeresConfig &config);
    void optimizerLoop();
  private:
    
    std::mutex mutex_;
    std::condition_variable cv_;
    std::shared_ptr<McGradient> mc_gr_;
    
    LBFGSBParam<float> param_;  // New parameter class
    LBFGSBSolver<float> solver_;  // New solver class
    
    Eigen::VectorXf rotations_=Eigen::VectorXf::Constant(3,1e-3); 
    // Bounds
    Eigen::Vector3f lb_ = {-M_PI,-6.0*2*M_PI,-M_PI};
    Eigen::Vector3f ub_ = {M_PI,0,M_PI};
    // int height_ = 400;
    // int width_ = 400;
    // float fx_ = 3.22418800e+03, fy_ = 3.21510040e+03, cx_ = (8.80357033e+02)-440, cy_ = (4.17066114e+02)-160 ; // dvx micro
    // int x_offset_=-440;
    // int y_offset_=-160;
    int idx=0;
    std::shared_ptr<std::ofstream> outfile_=NULL;
    int height_ = 720;
    int width_ = 1280;
    float fx_ = 3.22418800e+03, fy_ = 3.21510040e+03, cx_ = (8.80357033e+02), cy_ = (4.17066114e+02) ; // dvx micro
    uint64_t prev_t_=0;
    int events_=0;
    // double rotations_[3] = {1};
    TjhandleUniquePtr tjhandle_;
    int packet_count_ = 0;
    ros::Publisher compensated_image_pub_;
    ros::Publisher velocity_pub_;
    std::string ns;
    ros::NodeHandle nh_;
    // float t_new_frame_=-1;
    
    int new_height_ = 720;
    int new_width_ = 1280;
    int new_x_offset_=0;
    int new_y_offset_=0;
    bool new_config_=false;
    ros::Time whole_begin_=ros::Time::UNINITIALIZED;
  };

  class DvxplorerMcCeres
  {
  public:
    DvxplorerMcCeres(ros::NodeHandle &nh, ros::NodeHandle nh_private);
    ~DvxplorerMcCeres();

    void dataStop();

    static void onDisconnectUSB(void *);

  private:
		void callback(dvxplorer_mc_ceres::DVXplorer_MC_CeresConfig &config, uint32_t level);
    ros::Publisher compensated_image_pub_;
    std::string ns;
    ros::NodeHandle nh_;
    ros::Subscriber event_struct_sub_;
    // void eventStructCallback(const dvs_msgs::EventStruct::ConstPtr &msg);
    void eventStructCallback(const event_camera_codecs::EventPacketConstSharedPtr &msg);
    std::shared_ptr<EventProcessor> processor_;
    event_camera_codecs::DecoderFactory<EventPacket, EventProcessor> decoderFactory_;
		boost::shared_ptr<dynamic_reconfigure::Server<dvxplorer_mc_ceres::DVXplorer_MC_CeresConfig>> server_;
		dynamic_reconfigure::Server<dvxplorer_mc_ceres::DVXplorer_MC_CeresConfig>::CallbackType
			dynamic_reconfigure_callback_;
  };

} // namespace dvxplorer_mc_ceres