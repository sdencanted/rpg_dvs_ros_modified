#include "dvxplorer_mc_ceres/mc_gradient_lbfgsb.h"

#include <fstream>
#include <iostream>

#include <jetson-utils/cudaMappedMemory.h>
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include <thrust/device_vector.h>

#include "dvxplorer_mc_ceres/utils.h"
#include "dvxplorer_mc_ceres/motion_compensation.cuh"

int getMax(int *values, int num_el)
{
    int *out;
    cudaMalloc(&out, sizeof(int));
    size_t temp_cub_temp_size;
    int *temp_storage = NULL;
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, values, out, num_el, cub::Max(), 0);
    cudaDeviceSynchronize();
    cudaMalloc(&temp_storage, temp_cub_temp_size);
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, values, out, num_el, cub::Max(), 0);
    cudaDeviceSynchronize();
    int maximum;
    cudaMemcpy(&maximum, out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(out);
    cudaFree(temp_storage);
    return maximum;
}

float getMax(float *values, int num_el)
{
    float *out;
    cudaMalloc(&out, sizeof(float));
    size_t temp_cub_temp_size;
    float *temp_storage = NULL;
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, values, out, num_el, cub::Max(), 0);
    cudaDeviceSynchronize();
    cudaMalloc(&temp_storage, temp_cub_temp_size);
    cub::DeviceReduce::Reduce(temp_storage, temp_cub_temp_size, values, out, num_el, cub::Max(), 0);
    cudaDeviceSynchronize();
    float maximum;
    cudaMemcpy(&maximum, out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(out);
    cudaFree(temp_storage);
    return maximum;
}

float thrustMean(float *values, int num_el)
{
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(values);
    float sum1 = thrust::reduce(dev_ptr, dev_ptr + num_el, 0.0, thrust::plus<float>());
    return sum1 / num_el;
}

McGradient::~McGradient()
{
}
McGradient::McGradient(const float fx, const float fy, const float cx, const float cy, const int height, const int width) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), height_(height), width_(width)
{
    // cv_ = std::make_shared<std::condition_variable>();

    cudaStreamCreate(&stream_[0]);
    cudaStreamCreate(&stream_[1]);

    calculateBlockGridSizes();
    allocateImageRelated_();
    checkCudaErrors(cudaMalloc(&means_, 4 * sizeof(float)));
}
void McGradient::allocateImageRelated_()
{
    // create pinned memory for x,y,t,image,image dels
    checkCudaErrors(cudaMalloc(&image_and_jacobian_images_buffer_, (height_) * (width_) * sizeof(float) * 4));
    checkCudaErrors(cudaMalloc(&kronecker_image_, (height_) * (width_) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&bilinear_image_, (height_) * (width_) * sizeof(float)));
    checkCudaErrors(cudaMallocHost(&output_image, (height_) * (width_) * sizeof(uint8_t)));
    checkCudaErrors(cudaMallocHost((void **)&contrast_block_sum_, block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").second * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&contrast_del_x_block_sum_, block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").second * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&contrast_del_y_block_sum_, block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").second * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&contrast_del_z_block_sum_, block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").second * sizeof(float)));

    cudaMemsetAsync(image_and_jacobian_images_buffer_, 0, (height_) * (width_) * sizeof(float) * 4);
    cudaMemsetAsync(contrast_block_sum_, 0, block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").second * sizeof(float));
    cudaMemsetAsync(contrast_del_x_block_sum_, 0, block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").second * sizeof(float));
    cudaMemsetAsync(contrast_del_y_block_sum_, 0, block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").second * sizeof(float));
    cudaMemsetAsync(contrast_del_z_block_sum_, 0, block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").second * sizeof(float));
}
void McGradient::tryCudaAllocMapped(float **ptr, size_t size, std::string ptr_name)
{
    std::cout << "allocating cuda mem for " << ptr_name << std::endl;
    if (!cudaAllocMapped(ptr, size))
    {
        std::cout << "could not allocate cuda mem for " << ptr_name << std::endl;
    }
}
void McGradient::reset(int new_height, int new_width, int new_x_offset, int new_y_offset)
{
    if (new_height > 0 && new_width > 0)
    {
        height_ = new_height;
        width_ = new_width;
        calculateBlockGridSizes();
    }
    x_offset_ = new_x_offset;
    y_offset_ = new_y_offset;
    checkCudaErrors(cudaFree(image_and_jacobian_images_buffer_));
    checkCudaErrors(cudaFree(kronecker_image_));
    checkCudaErrors(cudaFree(bilinear_image_));
    checkCudaErrors(cudaFreeHost(output_image));
    checkCudaErrors(cudaFreeHost(contrast_block_sum_));
    checkCudaErrors(cudaFree(contrast_del_x_block_sum_));
    checkCudaErrors(cudaFree(contrast_del_y_block_sum_));
    checkCudaErrors(cudaFree(contrast_del_z_block_sum_));
    allocateImageRelated_();
}
void McGradient::allocate()
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
bool McGradient::Evaluate(const double *const parameters,
                          double *residuals,
                          double *gradient)
{

    motionCompensateAndFillImageGaussian5_(parameters);

    // ROS_INFO("thrust contrast mean %f",thrustMean(image_and_jacobian_images_buffer_,height_,width_));
    getContrastDelBatchReduce_(residuals, gradient);

    // ROS_INFO("results for iter %d rot %f %f %f con %f grad %f %f %f", iterations,  parameters[0], parameters[1], parameters[2], residuals[0], gradient[0], gradient[1], gradient[2]);
    // std::cout<<"results for iter "<<iterations<< "rot "<<parameters[0]<<" "<<parameters[1]<<" "<<parameters[2]<<" con";
    // std::cout<<residuals[0]<<" grads "<<gradient[0]<<" "<<gradient[1]<<" "<<gradient[2]<<std::endl;

    return true;
}
bool McGradient::EvaluateBilinear(const double *const parameters,
                                  double *residuals,
                                  double *gradient)
{
    // std::cout<<num_events_to_be_considered_<<std::endl;
    motionCompensateAndFillImageBilinear_(parameters);

    getContrastDelBatchReduce_(residuals, gradient);

    // ROS_INFO("results  for bilinear iter %d rot %f %f %f con %f grad %f %f %f", iterations, parameters[0], parameters[1], parameters[2], residuals[0], gradient[0], gradient[1], gradient[2]);
    // std::cout<<"results for iter "<<iterations<< "rot "<<parameters[0]<<" "<<parameters[1]<<" "<<parameters[2]<<" con";
    // std::cout<<residuals[0]<<" grads "<<gradient[0]<<" "<<gradient[1]<<" "<<gradient[2]<<std::endl;

    return true;
}
void McGradient::clearImages()
{
    cudaMemsetAsync(kronecker_image_, 0, sizeof(int) * width_ * height_, stream_[1]);
    cudaMemsetAsync(bilinear_image_, 0, sizeof(int) * width_ * height_, stream_[1]);
    // cudaMemsetAsync(image_and_jacobian_images_buffer_,0,sizeof(int)*width_*height_*4,stream_[1]);
}
void McGradient::syncClearImages()
{
    cudaStreamSynchronize(stream_[1]);
}
int64_t McGradient::GetApproxMiddleT()
{
    return approx_middle_t_;
}
int64_t McGradient::GetActualMiddleT()
{
    return actual_middle_t_;
}
int64_t McGradient::GetApproxLastT()
{

    return approx_last_t_;
}
int64_t McGradient::GetActualLastT()
{
    return actual_last_t_;
}
void McGradient::setOffsets(int x_offset, int y_offset)
{
    x_offset_ = x_offset;
    y_offset_ = y_offset;
}
void McGradient::AddData(uint64_t t, uint16_t x, uint16_t y)
{
    if (first_event_)
    {
        SetTimestampGoals(t + 10000000 - t % 10000000);
    }
    else if (!reached_middle_t_ && t > approx_middle_t_)
    {

        // too little events
        if (num_events_to_be_considered_ < min_num_events_ / 2)
        {

            // ROS_INFO("too little events before midpoint: %d/%d", num_events_to_be_considered_, min_num_events_ / 2);
            ClearEvents();
            SetTimestampGoals(GetApproxMiddleTs() + (int64_t)10000000);
            return;
        }
        else
        {
            reached_middle_t_ = true;
            actual_middle_t_ = t;
            // ROS_INFO("reached middle t at %d idx, approx %ld actual %ld",num_events_to_be_considered_,approx_middle_t_,actual_middle_t_);

            // capture first
            middle_t_first_event_idx_ = num_events_to_be_considered_;
        }
    }
    else if (!after_middle_t_ && reached_middle_t_ && t > actual_middle_t_)
    {
        // ROS_INFO("left true middle t %ld at %d idx, t=%ld",actual_middle_t_,num_events_to_be_considered_,t);
        after_middle_t_ = true;
        int middle_t_last_event_idx = num_events_to_be_considered_;
        int middle_idx = (middle_t_first_event_idx_ + middle_t_last_event_idx) / 2;

        final_event_idx_ = middle_idx + target_num_events_ / 2;
    }
    t_[num_events_to_be_considered_ % target_num_events_] = ((int64_t)t - approx_middle_t_) / 1e9;

    // ROS_INFO("%f %ld %ld",t_[num_events_to_be_considered_],t,evk_middle_t_);
    x_unprojected_[num_events_to_be_considered_ % target_num_events_] = (x - cx_) / fx_;
    y_unprojected_[num_events_to_be_considered_ % target_num_events_] = (y - cy_) / fy_;
    x_[num_events_to_be_considered_ % target_num_events_] = x;
    y_[num_events_to_be_considered_ % target_num_events_] = y;
    num_events_to_be_considered_++;
}
bool McGradient::ReadyToMC(uint64_t t)
{
    if (after_middle_t_ && (num_events_to_be_considered_ > final_event_idx_ || t >= approx_last_t_))
    {
        actual_last_t_ = t;
        // ROS_INFO("ready to MC, t=%f, idx=%d",t_[(num_events_to_be_considered_%target_num_events_) -1] ,num_events_to_be_considered_ );
        return true;
    }
    return false;
}
bool McGradient::SufficientEvents()
{
    return num_events_to_be_considered_ > min_num_events_;
}
void McGradient::ClearEvents()
{
    iterations = 0;
    num_events_to_be_considered_ = 0;
    reached_middle_t_ = false;
    after_middle_t_ = false;
}
void McGradient::SetTimestampGoals(int64_t new_approx_middle_t_)
{
    approx_middle_t_ = new_approx_middle_t_;
    approx_last_t_ = approx_middle_t_ + (int64_t)5000000;
    first_event_ = false;
}
int64_t McGradient::GetApproxMiddleTs()
{
    return approx_middle_t_;
}
void McGradient::SumImage()
{
    std::cout << thrustMean(image_and_jacobian_images_buffer_, height_ * width_) << std::endl;
}
void McGradient::GenerateImage(const double *const ang_vels)
{
    motionCompensate_(ang_vels);
    fillImageKroneckerNoJacobians_(true);
    int maximum = getMax(kronecker_image_, height_ * width_);
    rescaleIntensityInt_(kronecker_image_, maximum);
};
void McGradient::GenerateUncompensatedImage()
{
    fillImageKroneckerNoJacobians_(false);
    int maximum = getMax(kronecker_image_, height_ * width_);
    rescaleIntensityInt_(kronecker_image_, maximum);
};
void McGradient::GenerateImageBilinear(const double *const ang_vels)
{
    motionCompensate_(ang_vels);
    motionCompensateAndFillImageBilinear_(ang_vels);
    float maximum = getMax(bilinear_image_, height_ * width_);
    rescaleIntensityFloat_(bilinear_image_, maximum);
};
void McGradient::GenerateUncompensatedImageBilinear()
{
    cudaMemset(bilinear_image_, 0, height_ * width_ * sizeof(float));
    fillImageBilinearNoJacobians_(false);
    float maximum = getMax(bilinear_image_, height_ * width_);
    rescaleIntensityFloat_(bilinear_image_, maximum);
};
float McGradient::operator()(const VectorXf &x, VectorXf &grad)
{
    double fx = 0.0;
    double ang_vels[3];
    double gradients[3];
    for (int i = 0; i < 3; i++)
    {
        ang_vels[i] = x[i];
        gradients[i] = grad[i];
    }
    ang_vels[0] = x[0];
    if (use_bilinear_)
    {

        EvaluateBilinear(ang_vels,
                         &fx,
                         gradients);
    }
    else
    {

        Evaluate(ang_vels,
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
void McGradient::setUseBilinear(bool use)
{
    use_bilinear_ = use;
}
int McGradient::getNumEventsToBeConsidered()
{
    return num_events_to_be_considered_;
}

int McGradient::numEventsInWindow()
{
    return std::min(target_num_events_, num_events_to_be_considered_);
}
void McGradient::motionCompensateAndFillImageGaussian5_(const double *const ang_vels)
{
    prev_fill_ = "gaussian5";
    int smemSize = block_grid_sizes_["motionCompensateAndFillImageGaussian5"].first * sizeof(float);
    g_motionCompensateAndFillImageGaussian5<<<block_grid_sizes_["motionCompensateAndFillImageGaussian5"].second, block_grid_sizes_["motionCompensateAndFillImageGaussian5"].first, smemSize, stream_[0]>>>(fx_, fy_, cx_, cy_, height_, width_, numEventsInWindow(), x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_and_jacobian_images_buffer_, ang_vels[0], ang_vels[1], ang_vels[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, x_offset_, y_offset_);
}
void McGradient::motionCompensateAndFillImageBilinear_(const double *const ang_vels)
{
    prev_fill_ = "bilinear";
    int smemSize = block_grid_sizes_["motionCompensateAndFillImageBilinear"].first * sizeof(float);
    g_motionCompensateAndFillImageBilinear<<<block_grid_sizes_["motionCompensateAndFillImageBilinear"].second, block_grid_sizes_["motionCompensateAndFillImageBilinear"].first, smemSize, stream_[0]>>>(fx_, fy_, cx_, cy_, height_, width_, numEventsInWindow(), x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, image_and_jacobian_images_buffer_, ang_vels[0], ang_vels[1], ang_vels[2], contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, x_offset_, y_offset_);
}
void McGradient::motionCompensate_(const double *const ang_vels)
{
    int smemSize = block_grid_sizes_["motionCompensate"].first * sizeof(float);
    g_motionCompensate<<<block_grid_sizes_["motionCompensate"].second, block_grid_sizes_["motionCompensate"].first, smemSize>>>(fx_, fy_, cx_, cy_, height_, width_, numEventsInWindow(), x_unprojected_, y_unprojected_, x_prime_, y_prime_, t_, ang_vels[0], ang_vels[1], ang_vels[2], x_offset_, y_offset_);
}

void McGradient::fillImageKroneckerNoJacobians_(bool use_motion_compensated_vals)
{
    if (use_motion_compensated_vals)
    {
        g_fillImageKroneckerNoJacobians<<<block_grid_sizes_["fillImageKroneckerNoJacobians"].second, block_grid_sizes_["fillImageKroneckerNoJacobians"].first>>>(height_, width_, numEventsInWindow(), x_prime_, y_prime_, kronecker_image_);
    }
    else
    {
        g_fillImageKroneckerNoJacobians<<<block_grid_sizes_["fillImageKroneckerNoJacobians"].second, block_grid_sizes_["fillImageKroneckerNoJacobians"].first>>>(height_, width_, numEventsInWindow(), x_, y_, kronecker_image_);
    }
}
void McGradient::fillImageBilinearNoJacobians_(bool use_motion_compensated_vals)
{
    if (use_motion_compensated_vals)
    {
        g_fillImageBilinearNoJacobians<<<block_grid_sizes_["fillImageBilinearNoJacobians"].second, block_grid_sizes_["fillImageBilinearNoJacobians"].first>>>(height_, width_, numEventsInWindow(), x_prime_, y_prime_, bilinear_image_);
    }
    else
    {
        g_fillImageBilinearNoJacobians<<<block_grid_sizes_["fillImageBilinearNoJacobians"].second, block_grid_sizes_["fillImageBilinearNoJacobians"].first>>>(height_, width_, numEventsInWindow(), x_, y_, bilinear_image_);
    }
}
void McGradient::rescaleIntensityFloat_(float *image, float maximum)
{

    g_rescaleIntensityFloat<<<block_grid_sizes_["rescaleIntensityFloat"].second, block_grid_sizes_["rescaleIntensityFloat"].first>>>(image, output_image, maximum, height_ * width_);
    cudaDeviceSynchronize();
}
void McGradient::rescaleIntensityInt_(int *image, int maximum)
{

    g_rescaleIntensityInt<<<block_grid_sizes_["rescaleIntensityInt"].second, block_grid_sizes_["rescaleIntensityInt"].first>>>(image, output_image, maximum, height_ * width_);
    cudaDeviceSynchronize();
}
void McGradient::getContrastDelBatchReduce_(double *residuals, double *gradient)
{
    int prev_gridsize = 0;
    if (prev_fill_ == "gaussian5")
    {
        prev_gridsize = block_grid_sizes_["motionCompensateAndFillImageGaussian5"].second;
    }
    else if (prev_fill_ == "bilinear")
    {
        prev_gridsize = block_grid_sizes_["motionCompensateAndFillImageBilinear"].second;
    }

    int smemSize = (block_grid_sizes_["calculateAndReduceContrastAndJacobians"].first <= 32) ? 2 * block_grid_sizes_["calculateAndReduceContrastAndJacobians"].first * sizeof(float) : block_grid_sizes_["calculateAndReduceContrastAndJacobians"].first * sizeof(float);

    int image_pixels = height_ * width_;
    void *kernel_args[] = {
        (void *)&image_and_jacobian_images_buffer_,
        (void *)&image_pixels,
        (void *)&means_,
        (void *)&contrast_block_sum_,
        (void *)&contrast_del_x_block_sum_,
        (void *)&contrast_del_y_block_sum_,
        (void *)&contrast_del_z_block_sum_,
        (void *)&prev_gridsize,
    };

    switch (block_grid_sizes_["calculateAndReduceContrastAndJacobians"].first)
    {
    case 512:
        cudaLaunchCooperativeKernel((void *)g_calculateAndReduceContrastAndJacobians<512>, block_grid_sizes_["calculateAndReduceContrastAndJacobians"].second, 512, kernel_args, smemSize, stream_[0]);
        break;
    case 256:
        cudaLaunchCooperativeKernel((void *)g_calculateAndReduceContrastAndJacobians<256>, block_grid_sizes_["calculateAndReduceContrastAndJacobians"].second, 256, kernel_args, smemSize, stream_[0]);
        break;
    case 128:
        cudaLaunchCooperativeKernel((void *)g_calculateAndReduceContrastAndJacobians<128>, block_grid_sizes_["calculateAndReduceContrastAndJacobians"].second, 128, kernel_args, smemSize, stream_[0]);
        break;
    case 64:
        cudaLaunchCooperativeKernel((void *)g_calculateAndReduceContrastAndJacobians<64>, block_grid_sizes_["calculateAndReduceContrastAndJacobians"].second, 64, kernel_args, smemSize, stream_[0]);
        break;
    default:
        break;
    }
    cudaStreamSynchronize(stream_[0]);
    checkCudaErrors(cudaPeekAtLastError());

    prev_gridsize = block_grid_sizes_["calculateAndReduceContrastAndJacobians"].second;
    if (prev_gridsize > 256)
    {
        g_reduceContrastAndJacobiansPt2<512><<<4, 128, 128 * sizeof(float), stream_[0]>>>(contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, prev_gridsize);
    }

    else if (prev_gridsize > 128)
    {
        g_reduceContrastAndJacobiansPt2<256><<<4, 128, 128 * sizeof(float), stream_[0]>>>(contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, prev_gridsize);
    }

    else if (prev_gridsize > 64)
    {
        g_reduceContrastAndJacobiansPt2<128><<<4, 128, 128 * sizeof(float), stream_[0]>>>(contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, prev_gridsize);
    }

    else
    {
        g_reduceContrastAndJacobiansPt2<64><<<4, 128, 128 * sizeof(float), stream_[0]>>>(contrast_block_sum_, contrast_del_x_block_sum_, contrast_del_y_block_sum_, contrast_del_z_block_sum_, prev_gridsize);
    }

    // checkCudaErrors(cudaPeekAtLastError());
    cudaMemsetAsync(image_and_jacobian_images_buffer_, 0, image_pixels * sizeof(float) * 4, stream_[1]);
    // checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaStreamSynchronize(stream_[0]));
    checkCudaErrors(cudaStreamSynchronize(stream_[1]));
    // cudaDeviceSynchronize();
    {
        // nvtx3::scoped_range r{"final contrast"};
        residuals[0] = -contrast_block_sum_[0] / image_pixels;
        gradient[0] = -2 * contrast_block_sum_[1] / image_pixels;
        gradient[1] = -2 * contrast_block_sum_[2] / image_pixels;
        gradient[2] = -2 * contrast_block_sum_[3] / image_pixels;
    }
}

void McGradient::calculateBlockGridSizes()
{
    block_grid_sizes_.clear();

    // temp variables
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
    // maximum occupancy for a full device launch
    int gridSize; // The actual grid size needed, based on input size

    blockSize=512; // fixed to account for reduction of 4 variables (1 contrast 3 jacobians)
    gridSize = (target_num_events_ + blockSize - 1) / blockSize;
    block_grid_sizes_.insert(std::make_pair(std::string("motionCompensateAndFillImageGaussian5"), std::make_pair(blockSize, gridSize)));

    blockSize=512; // fixed to account for reduction of 4 variables (1 contrast 3 jacobians)
    gridSize = (target_num_events_ + blockSize - 1) / blockSize;
    block_grid_sizes_.insert(std::make_pair(std::string("motionCompensateAndFillImageBilinear"), std::make_pair(blockSize, gridSize)));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       g_motionCompensate, 0, 0);
    gridSize = (target_num_events_ + blockSize - 1) / blockSize;
    block_grid_sizes_.insert(std::make_pair(std::string("motionCompensate"), std::make_pair(blockSize, gridSize)));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       g_fillImageKroneckerNoJacobians, 0, 0);
    gridSize = (target_num_events_ + blockSize - 1) / blockSize;
    block_grid_sizes_.insert(std::make_pair(std::string("fillImageKroneckerNoJacobians"), std::make_pair(blockSize, gridSize)));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       g_fillImageBilinearNoJacobians, 0, 0);
    gridSize = (target_num_events_ + blockSize - 1) / blockSize;
    block_grid_sizes_.insert(std::make_pair(std::string("fillImageBilinearNoJacobians"), std::make_pair(blockSize, gridSize)));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       g_rescaleIntensityFloat, 0, 0);
    gridSize = (target_num_events_ + blockSize - 1) / blockSize;
    block_grid_sizes_.insert(std::make_pair(std::string("rescaleIntensityFloat"), std::make_pair(blockSize, gridSize)));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       g_rescaleIntensityInt, 0, 0);
    gridSize = (target_num_events_ + blockSize - 1) / blockSize;
    block_grid_sizes_.insert(std::make_pair(std::string("rescaleIntensityInt"), std::make_pair(blockSize, gridSize)));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       g_motionCompensateAndFillImageGaussian5, 0, 0);
    gridSize = (target_num_events_ + blockSize - 1) / blockSize;
    block_grid_sizes_.insert(std::make_pair(std::string("motionCompensateAndFillImageGaussian5"), std::make_pair(blockSize, gridSize)));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       g_motionCompensateAndFillImageBilinear, 0, 0);
    gridSize = (target_num_events_ + blockSize - 1) / blockSize;
    block_grid_sizes_.insert(std::make_pair(std::string("motionCompensateAndFillImageBilinear"), std::make_pair(blockSize, gridSize)));

    // cooperative groups special calculation
    int numBlocksPerSm = 0;
    // Number of threads my_kernel will be launched with
    cudaDeviceProp deviceProp;
    int dev = 0;
    cudaGetDeviceProperties(&deviceProp, dev);

    for (int desired_block_size = 512; desired_block_size >= 64; desired_block_size >> 1)
    {
        switch (desired_block_size)
        {
        case 512:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                               g_calculateAndReduceContrastAndJacobians<512>, 0, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, g_calculateAndReduceContrastAndJacobians<512>, desired_block_size, 0);
            break;
        case 256:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                               g_calculateAndReduceContrastAndJacobians<256>, 0, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, g_calculateAndReduceContrastAndJacobians<256>, desired_block_size, 0);
            break;
        case 128:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                               g_calculateAndReduceContrastAndJacobians<128>, 0, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, g_calculateAndReduceContrastAndJacobians<128>, desired_block_size, 0);
            break;
        case 64:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                               g_calculateAndReduceContrastAndJacobians<64>, 0, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, g_calculateAndReduceContrastAndJacobians<64>, desired_block_size, 0);
            break;
        default:
            break;
        }
        if (blockSize >= desired_block_size)
        {
            gridSize = deviceProp.multiProcessorCount * numBlocksPerSm;
            block_grid_sizes_.insert(std::make_pair(std::string("calculateAndReduceContrastAndJacobians"), std::make_pair(desired_block_size, gridSize)));
            // std::cout << "block size " << block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").first << " grid size " << block_grid_sizes_.at("calculateAndReduceContrastAndJacobians").second << " min grid size " << minGridSize << std::endl;
            if (gridSize < minGridSize)
            {
                std::cout << "warning: gridSize " << gridSize << " insufficient for max occupancy (" << minGridSize << ")" << std::endl;
            }

            break;
        }
        else if (desired_block_size == 64)
        {
            std::cerr << "smallest block size 64 insufficient for calculateAndReduceContrastAndJacobians" << std::endl;
        }
    }
}