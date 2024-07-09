#ifndef MC_GRADIENT_LBFGSPP_H
#define MC_GRADIENT_LBFGSPP_H

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <sys/resource.h>
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Core>

using Eigen::VectorXf;

int getMax(int *values, int num_el);
float getMax(float *values, int num_el);
float thrustMean(float *values, int num_el);

class McGradient
{

public:
    ~McGradient();
    McGradient(const float fx, const float fy, const float cx, const float cy, const int height, const int width);
    void allocateImageRelated_();
    void tryCudaAllocMapped(float **ptr, size_t size, std::string ptr_name);
    void reset(int new_height = 0, int new_width = 0, int new_x_offset = 0, int new_y_offset = 0);
    void allocate();
    bool Evaluate(const double *const parameters,
                  double *residuals,
                  double *gradient) ;
    bool EvaluateBilinear(const double *const parameters,
                          double *residuals,
                          double *gradient) ;
    void clearImages();
    void syncClearImages();
    int64_t GetApproxMiddleT();
    int64_t GetActualMiddleT();
    int64_t GetApproxLastT();
    int64_t GetActualLastT();
    void setOffsets(int x_offset, int y_offset);
    void AddData(uint64_t t, uint16_t x, uint16_t y);
    bool ReadyToMC(uint64_t t);
    bool SufficientEvents();
    void ClearEvents();
    void SetTimestampGoals(int64_t new_approx_middle_t_);
    int64_t GetApproxMiddleTs();
    void SumImage();
    void GenerateImage(const double *const ang_vels);
    void GenerateUncompensatedImage();
    void GenerateImageBilinear(const double *const ang_vels);
    void GenerateUncompensatedImageBilinear();
    // function called by minimize in LBFGSB (/src/driver_evk4_lbfgsb.cpp)
    float operator()(const VectorXf &x, VectorXf &grad);
    void setUseBilinear(bool use);

    // calculates block and grid sizes for all CUDA kernels
    void calculateBlockGridSizes();
    int getNumEventsToBeConsidered();
    int numEventsInWindow();
    int f_count = 0;
    int g_count = 0;

    bool reconfigure_pending = false;
    uint8_t *output_image;
    int iterations = 0;

private:
    void motionCompensateAndFillImageGaussian5_(const double *const ang_vels);
    void motionCompensateAndFillImageBilinear_(const double *const ang_vels);
    void motionCompensate_(const double *const ang_vels);
    void getContrastDelBatchReduce_(double *residuals, double *gradient);
    void fillImageKroneckerNoJacobians_(bool use_motion_compensated_vals);
    void fillImageBilinearNoJacobians_(bool use_motion_compensated_vals);
    void rescaleIntensityFloat_(float *image,float maximum);
    void rescaleIntensityInt_(int *image, int maximum);

    std::string prev_fill_;
    float *x_unprojected_ = NULL;
    float *y_unprojected_ = NULL;
    float *x_ = NULL;
    float *y_ = NULL;
    float *x_prime_ = NULL;
    float *y_prime_ = NULL;
    float *t_ = NULL;
    int height_;
    int width_;
    int num_events_to_be_considered_ = 0;
    int target_num_events_ = 30000;
    int min_num_events_ = 10000;
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
    float *bilinear_image_;
    std::shared_ptr<std::thread> memset_thread_;
    std::mutex m_;
    std::shared_ptr<std::condition_variable> cv_;
    bool running = true;
    cudaStream_t stream_[2];
    bool allocated_ = false;
    int64_t approx_middle_t_ = 0;
    int64_t approx_last_t_ = 0;
    int64_t actual_middle_t_ = 0;
    int64_t actual_last_t_ = 0;
    int middle_t_first_event_idx_ = 0;
    int final_event_idx_ = 0;
    int x_offset_ = 0;
    int y_offset_ = 0;
    std::map<std::string, std::pair<int, int>> block_grid_sizes_;
};

#endif // MC_GRADIENT_LBFGSPP_H