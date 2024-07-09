
#ifndef MOTION_COMPENSATION_H
#define MOTION_COMPENSATION_H
#include <cstdint>
__global__ void g_motionCompensate(float fx, float fy, float cx, float cy, int height, int width, int num_events, const float *x_unprojected, const float *y_unprojected, float *x_prime, float *y_prime, float *t, const float rotation_x, const float rotation_y, const float rotation_z, int x_offset, int y_offset);
__global__ void g_motionCompensateAndFillImageGaussian5(float fx, float fy, float cx, float cy, int height, int width, int num_events, const float *x_unprojected, const float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int x_offset, int y_offset);
__global__ void g_motionCompensateAndFillImageBilinear(float fx, float fy, float cx, float cy, int height, int width, int num_events, const float *x_unprojected, const float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int x_offset, int y_offset);
__global__ void g_fillImageKroneckerNoJacobians(int height, int width, int num_events, float *x_prime, float *y_prime, int *image);
__global__ void g_fillImageBilinearNoJacobians(int height, int width, int num_events, float *x_prime, float *y_prime, float *image);
template <int block_size>
__global__ void g_calculateAndReduceContrastAndJacobians(float *image, int num_elements, float *means, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);
// 4 blocks x threads
template <int prev_block_size>
__global__ void g_reduceContrastAndJacobiansPt2(float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);
__global__ void g_rescaleIntensityInt(int *image, uint8_t *output_image, int maximum, int numel);
__global__ void g_rescaleIntensityFloat(float *image, uint8_t *output_image, float maximum, int numel);
#endif // MOTION_COMPENSATION_H