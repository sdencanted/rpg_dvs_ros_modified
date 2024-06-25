#ifndef MC_GRADIENT_H
#define MC_GRADIENT_H
#include "ceres/ceres.h"
#include "ceres/numeric_diff_options.h"
#include <dvs_msgs/EventStruct.h>
// #include "glog/logging.h"

// CUDA
#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "dvxplorer_mc_ceres/motion_compensation_float.h"
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
// A CostFunction implementing motion compensation then calculating contrast, as well as the jacobian.
class McGradient final : public ceres::FirstOrderFunction
{

public:
    ~McGradient()
    {
        // checkCudaErrors(cudaFreeHost(x_unprojected_));
        // checkCudaErrors(cudaFreeHost(y_unprojected_));
        // checkCudaErrors(cudaFree(x_prime_));
        // checkCudaErrors(cudaFree(y_prime_));
        // checkCudaErrors(cudaFreeHost(t_));
        // checkCudaErrors(cudaFreeHost(x_));
        // checkCudaErrors(cudaFreeHost(y_));
        // cudaStreamDestroy(stream_[0]);
        // cudaStreamDestroy(stream_[1]);

        // checkCudaErrors(cudaFreeHost(contrast_block_sum_));
        // checkCudaErrors(cudaFree(contrast_del_x_block_sum_));
        // checkCudaErrors(cudaFree(contrast_del_y_block_sum_));
        // checkCudaErrors(cudaFree(contrast_del_z_block_sum_));
        // checkCudaErrors(cudaFreeHost(means_));

        // // running = false;
        // // cv_->notify_one();
        // // if (memset_thread_->joinable())
        // //     memset_thread_->join();
    }
    McGradient(const float fx, const float fy, const float cx, const float cy, const int height, const int width) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(height), width_(width)
    {
        // cv_ = std::make_shared<std::condition_variable>();

        cudaStreamCreate(&stream_[0]);
        cudaStreamCreate(&stream_[1]);
        // create pinned memory for x,y,t,image,image dels

        checkCudaErrors(cudaMalloc(&image_and_jacobian_images_buffer_, (height_) * (width_) * sizeof(float) * 4));
        int gridSize = std::min(512, (height * width + 512 - 1) / 512);
        // checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum_, 128 * sizeof(float)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, 128 * sizeof(float)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, 128 * sizeof(float)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, 128 * sizeof(float)));
        checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, gridSize * sizeof(float)));
        checkCudaErrors(cudaMalloc(&means_, 4 * sizeof(float)));

        cudaMemsetAsync(image_and_jacobian_images_buffer_, 0, (height_) * (width_) * sizeof(float) * 4);
        cudaMemsetAsync(contrast_block_sum_, 0, gridSize * sizeof(float));
        cudaMemsetAsync(contrast_del_x_block_sum_, 0, gridSize * sizeof(float));
        cudaMemsetAsync(contrast_del_y_block_sum_, 0, gridSize * sizeof(float));
        cudaMemsetAsync(contrast_del_z_block_sum_, 0, gridSize * sizeof(float));

        // ReplaceData(x, y, t, num_events_);

        // for uncompensated image
    }
    void tryCudaAllocMapped(float **ptr, size_t size, std::string ptr_name)
    {
        std::cout << "allocating cuda mem for " << ptr_name << std::endl;
        if (!cudaAllocMapped(ptr, size))
        {
            std::cout << "could not allocate cuda mem for " << ptr_name << std::endl;
        }
    }
    void reset()
    {
        // cudaStreamDestroy(stream_[0]);
        // cudaStreamDestroy(stream_[1]);
        // cudaStreamCreate(&stream_[0]);
        // cudaStreamCreate(&stream_[1]);
        // create pinned memory for x,y,t,image,image dels
        // cudaFree(image_and_jacobian_images_buffer_);
        // checkCudaErrors(cudaMalloc(&image_and_jacobian_images_buffer_, (height_) * (width_) * sizeof(float) * 4));
        int gridSize = std::min(512, (height_ * width_ + 512 - 1) / 512);

        // cudaFreeHost(contrast_block_sum_);
        // cudaFree(contrast_del_x_block_sum_);
        // cudaFree(contrast_del_y_block_sum_);
        // cudaFree(contrast_del_z_block_sum_);
        // cudaFree(means_);
        // checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum_, gridSize * sizeof(float)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, gridSize * sizeof(float)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, gridSize * sizeof(float)));
        // checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, gridSize * sizeof(float)));
        // checkCudaErrors(cudaMalloc(&means_, 4 * sizeof(float)));

        cudaMemset(image_and_jacobian_images_buffer_, 0, (height_) * (width_) * sizeof(float) * 4);
        // cudaMemset(contrast_block_sum_, 0, gridSize * sizeof(float));
        // cudaMemset(contrast_del_x_block_sum_, 0, gridSize * sizeof(float));
        // cudaMemset(contrast_del_y_block_sum_, 0, gridSize * sizeof(float));
        // cudaMemset(contrast_del_z_block_sum_, 0, gridSize * sizeof(float));

        // ReplaceData(x, y, t, num_events_);

        // for uncompensated image
    }

    void deallocate()
    {
        checkCudaErrors(cudaFreeHost(x_unprojected_));
        checkCudaErrors(cudaFreeHost(y_unprojected_));
        checkCudaErrors(cudaFree(x_prime_));
        checkCudaErrors(cudaFree(y_prime_));
        checkCudaErrors(cudaFreeHost(t_));
        checkCudaErrors(cudaFreeHost(x_));
        checkCudaErrors(cudaFreeHost(y_));
    }
    void allocate()
    {
        checkCudaErrors(cudaMallocHost(&x_unprojected_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&y_unprojected_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMalloc(&x_prime_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMalloc(&y_prime_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&x_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&y_, target_num_events_ * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&t_, target_num_events_ * sizeof(float)));
        allocated_ = true;
    }
    void ReplaceData(std::vector<float> &x, std::vector<float> &y, std::vector<float> &t, const int num_events)
    {

        num_events_ = num_events;
        if (!allocated_ || target_num_events_ < num_events_)
        {
            target_num_events_ = std::max(num_events_, 30000);
            if (allocated_)
            {
                deallocate();
            }
            allocate();
        }
        cudaMemcpyAsync(x_, x.data(), num_events_ * sizeof(float), cudaMemcpyDefault);
        cudaMemcpyAsync(y_, y.data(), num_events_ * sizeof(float), cudaMemcpyDefault);
        // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
        // float scale=t[num_events-1]-t[0];
        float scale = 1e6;
        // find the middle t
        float middle_t = (t[num_events_ - 1] + t[0]) / 2;
        // precalculate unprojected x and y and store to x/y_unprojected (potentially redo in CUDA later on)
        for (int i = 0; i < num_events_; i++)
        {
            t_[i] = (t[i] - middle_t) / scale;
            x_unprojected_[i] = (x[i] - cx_) / fx_;
            y_unprojected_[i] = (y[i] - cy_) / fy_;
        }
    }

    void ReplaceData(std::vector<float> &x, std::vector<float> &y, std::vector<double> &t, const int num_events)
    {
        num_events_ = num_events;
        if (!allocated_ || target_num_events_ < num_events_)
        {
            target_num_events_ = std::max(num_events_, 30000);
            if (allocated_)
            {
                deallocate();
            }

            allocate();
        }
        cudaMemcpyAsync(x_, x.data(), num_events_ * sizeof(float), cudaMemcpyDefault);
        cudaMemcpyAsync(y_, y.data(), num_events_ * sizeof(float), cudaMemcpyDefault);
        // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
        // float scale=t[num_events-1]-t[0];
        // find the middle t
        double middle_t = (t.back() + t.front()) / 2;
        // precalculate unprojected x and y and store to x/y_unprojected (potentially redo in CUDA later on)
        for (int i = 0; i < num_events_; i++)
        {
            t_[i] = (t[i] - middle_t);
            x_unprojected_[i] = (x[i] - cx_) / fx_;
            y_unprojected_[i] = (y[i] - cy_) / fy_;
        }
        // std::cout<<"time "<<t_[0]<<" "<<t_[num_events_-1]<<std::endl;
    }

    void ReplaceData(const dvs_msgs::EventStruct::ConstPtr &msg)
    {
        num_events_ = msg->eventTime.data.size();
        if (!allocated_ || target_num_events_ < num_events_)
        {
            target_num_events_ = std::max(num_events_, 30000);
            if (allocated_)
            {
                deallocate();
            }
            allocate();
        }
        // precalculate tX-t0 and store to t (potentially redo in CUDA later on)
        // float scale=t[num_events-1]-t[0];
        float scale = 1e6;
        // find the middle t

        uint16_t middle_t = (msg->eventTime.data.back() + msg->eventTime.data.front()) / 2;
        // precalculate unprojected x and y and store to x/y_unprojected (potentially redo in CUDA later on)
        int i = 0;

        for (int i = 0; i < num_events_; i++)
        {
            t_[i] = (msg->eventTime.data[i] - middle_t) / scale;
            x_unprojected_[i] = (msg->eventArr.data[i * 2] - cx_) / fx_;
            y_unprojected_[i] = (msg->eventArr.data[i * 2 + 1] - cy_) / fy_;
        }
        // std::cout << num_events_ << " events loaded" << t_[0] << " " << t_[num_events_ - 1] << " " << msg->eventTime.data.back() << " " <<  msg->eventTime.data.front() << std::endl;
    }
    void AddData(uint64_t t, uint16_t x, uint16_t y)
    {
        if (first_event_)
        {
            first_event_ = false;
            approx_middle_t_ = t + 5 * 1e6;
        }
        else if (!reached_middle_t_ && t > approx_middle_t_)
        {
            reached_middle_t_ = true;
            actual_middle_t_ = t;

            // capture first
            middle_t_first_event_idx_ = num_events_;
        }
        else if (!after_middle_t_ && reached_middle_t_ && t > actual_middle_t_)
        {
            after_middle_t_ = true;
            final_event_idx_ = middle_t_first_event_idx_ + num_events_;

            // too little events
            if (final_event_idx_ < target_num_events_)
            {
                ClearEvents();
                return;
            }
        }
        t_[num_events_ % target_num_events_] = ((int64_t)t - approx_middle_t_) / 1e9;

        // ROS_INFO("%f %ld %ld",t_[num_events_],t,evk_middle_t_);
        x_unprojected_[num_events_ % target_num_events_] = (x - cx_) / fx_;
        y_unprojected_[num_events_ % target_num_events_] = (y - cy_) / fy_;
        num_events_++;
    }
    bool ReadyToMC()
    {
        if (after_middle_t_ && num_events_ > final_event_idx_)
        {
            return true;
        }
        return false;
    }
    void ClearEvents()
    {
        num_events_ = 0;
        first_event_ = true;
        reached_middle_t_ = false;
        after_middle_t_ = false;
        approx_middle_t_ += 1e5;
    }
    bool Evaluate(const double *const parameters,
                  double *residuals,
                  double *gradient) const override
    {
        fillImage(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_and_jacobian_images_buffer_, parameters[0], parameters[1], parameters[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_,stream_);

        getContrastDelBatchReduce(image_and_jacobian_images_buffer_, residuals, gradient, height_, width_,
                                  contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, means_, num_events_, stream_);
        // std::cout << "results for " << parameters[0] << " " << parameters[1] << " " << parameters[2] << "\n                                                                ";
        // std::cout << residuals[0] << " " << gradient[0] << " " << gradient[1] << " " << gradient[2] << std::endl;

        return true;
    }
    int NumParameters() const override { return 3; }
    void GenerateImage(const double *const rotations, uint8_t *output_image, float &contrast)
    {
        float *image;
        warpEvents(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, rotations[0], rotations[1], rotations[2]);
        cudaMallocHost(&image, sizeof(float) * height_ * width_);

        std::fill_n(image, (height_) * (width_), 0);
        fillImageKronecker(height_, width_, num_events_, x_prime_, y_prime_, image);
        // fillImageKronecker(height_, width_, num_events_, x_, y_, image);
        cudaDeviceSynchronize();
        float maximum = getMax(image, height_, width_) * 0.7;
        // std::cout<<maximum<<std::endl;

        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i * width_ + j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image[(i) * (width_) + j] / (maximum))));
            }
        }
        cudaFreeHost(image);
    };
    void GenerateUncompensatedImage(const double *const rotations, uint8_t *output_image, float &contrast)
    {
        float *image;
        cudaMallocHost(&image, sizeof(float) * height_ * width_);
        std::fill_n(image, (height_) * (width_), 0);
        fillImageKronecker(height_, width_, num_events_, x_, y_, image);
        cudaDeviceSynchronize();
        float maximum = getMax(image, height_, width_);
        // std::cout<<"un max "<<maximum<<std::endl;
        for (int i = 0; i < height_; i++)
        {
            for (int j = 0; j < width_; j++)
            {
                output_image[i * width_ + j] = (uint8_t)std::min(255.0, std::max(0.0, (255.0 * image[(i) * (width_) + j] / (maximum / 2))));
            }
        }
        cudaFreeHost(image);
    };
    void GenerateImageBilinear(const double *const rotations, uint8_t *output_image, float &contrast)
    {
        cudaMemsetAsync(image_and_jacobian_images_buffer_, 0, height_ * width_ * sizeof(float));
        cudaDeviceSynchronize();
        fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_and_jacobian_images_buffer_, rotations[0], rotations[1], rotations[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_);
        float *image;
        cudaMallocHost(&image, sizeof(float) * height_ * width_);
        cudaMemcpy(image, image_and_jacobian_images_buffer_, height_ * width_ * sizeof(float), cudaMemcpyDefault);
        float maximum = getMax(image_and_jacobian_images_buffer_, height_, width_);
        cudaMemsetAsync(image_and_jacobian_images_buffer_, 0, height_ * width_ * sizeof(float));
        float mean = thrustMean(image_and_jacobian_images_buffer_, height_, width_);
        // thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(image_and_jacobian_images_buffer_);
        // float sum1 = thrust::reduce(dev_ptr, dev_ptr+height_*width_, 0.0, thrust::plus<float>());
        // float mean= sum1/(height_*width_);

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
        fillImageBilinear(fx_, fy_, cx_, cy_, height_, width_, num_events_, x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_and_jacobian_images_buffer_, 0, 0, 0, contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_);
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

    int f_count = 0;
    int g_count = 0;

    // private:
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

    bool first_event_ = true;
    bool reached_middle_t_ = false;
    bool after_middle_t_ = false;

    float *image_and_jacobian_images_buffer_ = NULL;
    // float *image_del_theta_x_ = NULL;
    // float *image_del_theta_y_ = NULL;
    // float *image_del_theta_z_ = NULL;
    float fx_;
    float fy_;
    float cx_;
    float cy_;

    float *contrast_block_sum_;
    float *contrast_del_x_block_sum_;
    float *contrast_del_y_block_sum_;
    float *contrast_del_z_block_sum_;
    float *means_;
    std::shared_ptr<std::thread> memset_thread_;
    std::mutex m_;
    std::shared_ptr<std::condition_variable> cv_;
    bool running = true;
    cudaStream_t stream_[2];
    bool allocated_ = false;
    int64_t approx_middle_t_ = 0;
    int64_t actual_middle_t_ =0;
    int middle_t_first_event_idx_=0;
    int final_event_idx_=0;
};

class McGradientInterface final : public ceres::FirstOrderFunction
{
public:
    McGradientInterface(std::shared_ptr<McGradient> mc_gr) : mc_gr_(mc_gr)
    {
    }
    bool Evaluate(const double *const parameters,
                  double *residuals,
                  double *gradient) const override
    {
        return mc_gr_->Evaluate(parameters, residuals, gradient);
    }
    int NumParameters() const override { return 3; }

private:
    std::shared_ptr<McGradient> mc_gr_;
};
#endif // MC_GRADIENT_H