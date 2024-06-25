#ifndef MC_GRADIENT_LBFGSPP_H
#define MC_GRADIENT_LBFGSPP_H
// #include <dv-processing/core/frame.hpp>
// #include <dv-processing/io/mono_camera_recording.hpp>
// #include <dv-processing/core/multi_stream_slicer.hpp>
// #include "glog/logging.h"

// CUDA
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "motion_compensation_float.h"

#include "utils.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <pthread.h>
#include <sys/resource.h>
#include <thread>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/async/copy.h>
#include <Eigen/Core>
#include <LBFGSB.h> // Note the different header file

using Eigen::VectorXf;
class McGradient
{

public:
    ~McGradient()
    {
    }
    McGradient(const float fx, const float fy, const float cx, const float cy, const int height, const int width) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(height), width_(width)
    {
        // cv_ = std::make_shared<std::condition_variable>();

        cudaStreamCreate(&stream_[0]);
        cudaStreamCreate(&stream_[1]);

        allocateImageRelated_();
        checkCudaErrors(cudaMalloc(&means_, 4 * sizeof(float)));
    }
    void allocateImageRelated_()
    {
        // create pinned memory for x,y,t,image,image dels
        checkCudaErrors(cudaMalloc(&image_and_jacobian_images_buffer_, (height_) * (width_) * sizeof(float) * 4));
        checkCudaErrors(cudaMalloc(&kronecker_image_, (height_) * (width_) * sizeof(int)));
        checkCudaErrors(cudaMallocHost(&output_image, (height_) * (width_) * sizeof(uint8_t)));
        int gridSize = std::min(512, (height_ * width_ + 512 - 1) / 512);
        checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, gridSize * sizeof(float)));

        cudaMemsetAsync(image_and_jacobian_images_buffer_, 0, (height_) * (width_) * sizeof(float) * 4);
        cudaMemsetAsync(contrast_block_sum_, 0, gridSize * sizeof(float));
        cudaMemsetAsync(contrast_del_x_block_sum_, 0, gridSize * sizeof(float));
        cudaMemsetAsync(contrast_del_y_block_sum_, 0, gridSize * sizeof(float));
        cudaMemsetAsync(contrast_del_z_block_sum_, 0, gridSize * sizeof(float));
    }
    void tryCudaAllocMapped(float **ptr, size_t size, std::string ptr_name)
    {
        std::cout << "allocating cuda mem for " << ptr_name << std::endl;
        if (!cudaAllocMapped(ptr, size))
        {
            std::cout << "could not allocate cuda mem for " << ptr_name << std::endl;
        }
    }
    void reset(int new_height = 0, int new_width = 0, int new_x_offset = 0, int new_y_offset = 0)
    {
        if (new_height > 0 && new_width > 0)
        {
            height_ = new_height;
            width_ = new_width;
        }
        x_offset_ = new_x_offset;
        y_offset_ = new_y_offset;
        checkCudaErrors(cudaFree(image_and_jacobian_images_buffer_));
        checkCudaErrors(cudaFree(kronecker_image_));
        checkCudaErrors(cudaFreeHost(output_image));
        checkCudaErrors(cudaFreeHost(contrast_block_sum_));
        checkCudaErrors(cudaFree(contrast_del_x_block_sum_));
        checkCudaErrors(cudaFree(contrast_del_y_block_sum_));
        checkCudaErrors(cudaFree(contrast_del_z_block_sum_));
        allocateImageRelated_();
    }
    void allocate()
    {
        // ROS_INFO("allocating");
        checkCudaErrors(cudaMallocHost(&x_unprojected_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&y_unprojected_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMalloc(&x_prime_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMalloc(&y_prime_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&x_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&y_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&t_, target_num_events_ * sizeof(float)));
        allocated_ = true;
    }
    bool Evaluate(const double *const parameters,
                  double *residuals,
                  double *gradient) const
    {

        cudaMemset(image_and_jacobian_images_buffer_, 0, (height_) * (width_) * sizeof(float) * 4);
        int gridSize = std::min(512, (height_ * width_ + 512 - 1) / 512);
        cudaMemset(contrast_block_sum_, 0, gridSize * sizeof(float));
        cudaMemset(contrast_del_x_block_sum_, 0, gridSize * sizeof(float));
        cudaMemset(contrast_del_y_block_sum_, 0, gridSize * sizeof(float));
        cudaMemset(contrast_del_z_block_sum_, 0, gridSize * sizeof(float));
        cudaDeviceSynchronize();
        // std::cout<<num_events_<<std::endl;
        fillImage(fx_, fy_, cx_, cy_, height_, width_, std::min(target_num_events_, num_events_), x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_and_jacobian_images_buffer_, parameters[0], parameters[1], parameters[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, stream_, x_offset_, y_offset_);

        // ROS_INFO("thrust contrast mean %f",thrustMean(image_and_jacobian_images_buffer_,height_,width_));
        getContrastDelBatchReduce(image_and_jacobian_images_buffer_, residuals, gradient, height_, width_,
                                  contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, means_, std::min(target_num_events_, num_events_), stream_);

        // ROS_INFO("results for iter %d rot %f %f %f con %f grad %f %f %f", iterations,  parameters[0], parameters[1], parameters[2], residuals[0], gradient[0], gradient[1], gradient[2]);
        // std::cout<<"results for iter "<<iterations<< "rot "<<parameters[0]<<" "<<parameters[1]<<" "<<parameters[2]<<" con";
        // std::cout<<residuals[0]<<" grads "<<gradient[0]<<" "<<gradient[1]<<" "<<gradient[2]<<std::endl;

        return true;
    }
    bool EvaluateBilinear(const double *const parameters,
                          double *residuals,
                          double *gradient) const
    {
        // std::cout<<num_events_<<std::endl;
        fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, std::min(target_num_events_, num_events_), x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_and_jacobian_images_buffer_, parameters[0], parameters[1], parameters[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, stream_, x_offset_, y_offset_);

        getContrastDelBatchReduce(image_and_jacobian_images_buffer_, residuals, gradient, height_, width_,
                                  contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, means_, std::min(target_num_events_, num_events_), stream_);
        
        // artificially inflating gradients
        // residuals[0] *= 50;
        // for (int i = 0; i < 3; i++)
        // {
        //     gradient[i] *= 50;
        // }
        
        
        // ROS_INFO("results  for bilinear iter %d rot %f %f %f con %f grad %f %f %f", iterations, parameters[0], parameters[1], parameters[2], residuals[0], gradient[0], gradient[1], gradient[2]);
        // std::cout<<"results for iter "<<iterations<< "rot "<<parameters[0]<<" "<<parameters[1]<<" "<<parameters[2]<<" con";
        // std::cout<<residuals[0]<<" grads "<<gradient[0]<<" "<<gradient[1]<<" "<<gradient[2]<<std::endl;

        return true;
    }
    void clearImages()
    {
        cudaMemsetAsync(kronecker_image_, 0, sizeof(int) * width_ * height_, stream_[1]);
        // cudaMemsetAsync(image_and_jacobian_images_buffer_,0,sizeof(int)*width_*height_*4,stream_[1]);
    }
    void syncClearImages()
    {
        cudaStreamSynchronize(stream_[1]);
    }
    uint64_t GetMiddleT()
    {
        return approx_middle_t_;
    }
    void setOffsets(int x_offset, int y_offset)
    {
        x_offset_ = x_offset;
        y_offset_ = y_offset;
    }
    void AddData(uint64_t t, uint16_t x, uint16_t y)
    {
        if (first_event_)
        {
            SetTimestampGoals(t + 10000000 - t % 10000000);
        }
        else if (!reached_middle_t_ && t > approx_middle_t_)
        {
            reached_middle_t_ = true;
            actual_middle_t_ = t;
            // ROS_INFO("reached middle t at %d idx, approx %ld actual %ld",num_events_,approx_middle_t_,actual_middle_t_);

            // capture first
            middle_t_first_event_idx_ = num_events_;
        }
        else if (!after_middle_t_ && reached_middle_t_ && t > actual_middle_t_)
        {
            // ROS_INFO("left true middle t %ld at %d idx, t=%ld",actual_middle_t_,num_events_,t);
            after_middle_t_ = true;
            int middle_t_first_event_idx = num_events_;
            int middle_idx = (middle_t_first_event_idx + middle_t_first_event_idx_) / 2;

            // too little events
            if (middle_idx < target_num_events_ / 2)
            {

                // ROS_INFO("too little events before midpoint: %d/%d",middle_idx,target_num_events_/2);
                // ROS_INFO("few events before midpoint: %d/%d",middle_idx,target_num_events_/2);
                // ClearEvents();
                // return;
            }
            final_event_idx_ = middle_idx + target_num_events_ / 2;
            // final_event_idx_ = (middle_t_first_event_idx_ + num_events_);
        }
        t_[num_events_ % target_num_events_] = ((int64_t)t - approx_middle_t_) / 1e9;

        // ROS_INFO("%f %ld %ld",t_[num_events_],t,evk_middle_t_);
        x_unprojected_[num_events_ % target_num_events_] = (x - cx_) / fx_;
        y_unprojected_[num_events_ % target_num_events_] = (y - cy_) / fy_;
        x_[num_events_ % target_num_events_] = x;
        y_[num_events_ % target_num_events_] = y;
        num_events_++;
    }
    bool ReadyToMC(uint64_t t)
    {
        if (after_middle_t_ && (num_events_ > final_event_idx_ || t >= approx_last_t_))
        {

            // ROS_INFO("ready to MC, t=%f, idx=%d",t_[(num_events_%target_num_events_) -1] ,num_events_ );
            return true;
        }
        return false;
    }
    bool SufficientEvents()
    {
        return num_events_ > min_num_events_;
    }
    void ClearEvents()
    {
        iterations = 0;
        num_events_ = 0;
        reached_middle_t_ = false;
        after_middle_t_ = false;
    }
    void SetTimestampGoals(int64_t new_approx_middle_t_){
        approx_middle_t_ = new_approx_middle_t_;
        approx_last_t_ = approx_middle_t_ + 5000000;
        first_event_=false;
    }
    int64_t GetApproxMiddleTs(){
        return approx_middle_t_;
    }
    void SumImage()
    {
        std::cout << thrustMean(image_and_jacobian_images_buffer_, height_, width_) << std::endl;
    }
    void GenerateImage(const float *const rotations, float &contrast)
    {
        warpEvents(fx_, fy_, cx_, cy_, height_, width_, std::min(num_events_, target_num_events_), x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, rotations[0], rotations[1], rotations[2], x_offset_, y_offset_);
        fillImageKronecker(height_, width_, std::min(num_events_, target_num_events_), x_prime_, y_prime_, kronecker_image_);
        int maximum = getMax(kronecker_image_, height_, width_);
        rescaleIntensity(kronecker_image_, output_image, maximum, height_, width_);
    };
    void GenerateUncompensatedImage(float &contrast)
    {
        fillImageKronecker(height_, width_, std::min(num_events_, target_num_events_), x_, y_, kronecker_image_);
        int maximum = getMax(kronecker_image_, height_, width_);
        rescaleIntensity(kronecker_image_, output_image, maximum, height_, width_);
    };

    void GenerateImageBilinear(const double *const rotations, uint8_t *output_image, float &contrast)
    {
        cudaMemsetAsync(image_and_jacobian_images_buffer_, 0, height_ * width_ * sizeof(float));
        cudaDeviceSynchronize();
        fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_and_jacobian_images_buffer_, rotations[0], rotations[1], rotations[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, stream_, x_offset_, y_offset_);
        float *image;
        cudaMallocHost(&image, sizeof(float) * height_ * width_);
        cudaMemcpy(image, image_and_jacobian_images_buffer_, height_ * width_ * sizeof(float), cudaMemcpyDefault);
        float maximum = getMax(image_and_jacobian_images_buffer_, height_, width_);
        cudaMemsetAsync(image_and_jacobian_images_buffer_, 0, height_ * width_ * sizeof(float));
        float mean = thrustMean(image_and_jacobian_images_buffer_, height_, width_);
        float contrast_sum = 0;
        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i * width_ + j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image[(i) * (width_) + j] / (maximum / 2))));
                contrast_sum += (image[(i) * (width_) + j] - mean) * (image[(i) * (width_) + j] - mean);
            }
        }
        contrast = contrast_sum / (height_ * width_);
        cudaFreeHost(image);
    };
    void GenerateUncompensatedImageBilinear(const double *const rotations, uint8_t *output_image, float &contrast)
    {
        cudaDeviceSynchronize();
        fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_and_jacobian_images_buffer_, 0, 0, 0, contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, stream_, x_offset_, y_offset_);
        float *image;
        cudaMallocHost(&image, sizeof(float) * height_ * width_);
        cudaMemcpy(image, image_and_jacobian_images_buffer_, height_ * width_ * sizeof(float), cudaMemcpyDefault);
        float maximum = getMax(image_and_jacobian_images_buffer_, height_, width_);
        cudaMemsetAsync(image_and_jacobian_images_buffer_, 0, height_ * width_ * sizeof(float));
        float mean = thrustMean(image_and_jacobian_images_buffer_, height_, width_);
        float contrast_sum = 0;
        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i * width_ + j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image[(i) * (width_) + j] / (maximum / 2))));
                contrast_sum += (image[(i) * (width_) + j] - mean) * (image[(i) * (width_) + j] - mean);
            }
        }
        contrast = contrast_sum / (height_ * width_);
        cudaFreeHost(image);
    };
    float operator()(const VectorXf &x, VectorXf &grad)
    {
        double fx = 0.0;
        double rotations[3];
        double gradients[3];
        for (int i = 0; i < 3; i++)
        {
            rotations[i] = x[i];
            gradients[i] = grad[i];
        }
        rotations[0] = x[0];
        if (use_bilinear_)
        {

            EvaluateBilinear(rotations,
                             &fx,
                             gradients);
        }
        else
        {

            Evaluate(rotations,
                     &fx,
                     gradients);
        }

        for (int i = 0; i < 3; i++)
        {
            grad[i] = gradients[i];
        }
        iterations++;
        // std::cout<<fx<<std::endl;
        return (float)fx;
    }
    void setUseBilinear(bool use)
    {
        use_bilinear_ = use;
    }

    int f_count = 0;
    int g_count = 0;

    // private:
    uint8_t *output_image;
    thrust::device_vector<float> H;
    float *x_unprojected_ = NULL;
    float *y_unprojected_ = NULL;
    float *x_ = NULL;
    float *y_ = NULL;
    float *x_prime_ = NULL;
    float *y_prime_ = NULL;
    float *t_ = NULL;
    int height_;
    int width_;
    int num_events_ = 0;
    int target_num_events_ = 30000;
    int min_num_events_ = 1000;

    int iterations = 0;
    bool first_event_ = true;
    bool reached_middle_t_ = false;
    bool after_middle_t_ = false;
    bool use_bilinear_ = false;
    float *image_and_jacobian_images_buffer_ = NULL;
    float fx_;
    float fy_;
    float cx_;
    float cy_;

    float *contrast_block_sum_;
    float *contrast_del_x_block_sum_;
    float *contrast_del_y_block_sum_;
    float *contrast_del_z_block_sum_;
    float *means_;
    int *kronecker_image_;
    std::shared_ptr<std::thread> memset_thread_;
    std::mutex m_;
    std::shared_ptr<std::condition_variable> cv_;
    bool running = true;
    cudaStream_t stream_[2];
    bool allocated_ = false;
    int64_t approx_middle_t_ = 0;
    int64_t approx_last_t_ = 0;
    int64_t actual_middle_t_ = 0;
    int middle_t_first_event_idx_ = 0;
    int final_event_idx_ = 0;
    int x_offset_ = 0;
    int y_offset_ = 0;
};

#endif // MC_GRADIENT_LBFGSPP_H