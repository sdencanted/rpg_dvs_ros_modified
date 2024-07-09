
#ifndef MOTION_COMPENSATION_FLOAT_H
#define MOTION_COMPENSATION_FLOAT_H
#include <stdint.h>
#include <utility> //pair 
// #include <nvtx3/nvtx3.hpp>
void fillImage(std::pair<int,int>block_grid_sizes,float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z,  float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum,cudaStream_t const *stream, int x_offset=0,int y_offset=0);
void fillImageCalculate(std::pair<int,int>&block_grid_sizes,int max_num_events);
void motionCompensateAndFillImageBilinear(std::pair<int,int>block_grid_sizes,float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z,  float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum,cudaStream_t const *stream, int x_offset=0,int y_offset=0);
void motionCompensateAndFillImageBilinearCalculate(std::pair<int,int>&block_grid_sizes,int max_num_events);

void fillImageKroneckerNoJacobians(int height, int width, int num_events, float *x_prime, float *y_prime, int *image);

int getMax(int *image, int height, int width);   
float getMax(float *image, int height, int width);        

void getContrastDelBatchReduce(float *image, 
                               double *image_contrast, double *image_del_theta_contrast,
                               int height, int width,
                               float *contrast_block_sum,
                               float *contrast_del_x_block_sum,
                               float *contrast_del_y_block_sum,
                               float *contrast_del_z_block_sum,
                               float *means,
                               int num_events,
                               cudaStream_t const* stream);
void one_step_kernel(uint64_t seed, float* randoms, int numel);
float thrustMean(float* image_and_jacobian_images_buffer_,int height_,int width_);
void warpEvents(float fx, float fy, float cx, float cy, int height, int width, int num_events, float *x_unprojected, float *y_unprojected, float *x_prime, float *y_prime, float *t, const float rotation_x, const float rotation_y, const float rotation_z, int x_offset=0,int y_offset=0);
void rescaleIntensity(int* image,uint8_t* output_image,int maximum,int height,int width);
void rescaleIntensity(float* image,uint8_t* output_image,float maximum,int height,int width);
#endif // MOTION_COMPENSATION_H