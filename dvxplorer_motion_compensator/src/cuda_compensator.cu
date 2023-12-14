#include "dvxplorer_motion_compensator/cuda_compensator.h"

#include <cmath>
#include <algorithm> //for std::max
#include <cstdio>
#include <vector>

__global__ void motionCompensate_(uint16_t *exy, float *et, int32_t *mat_out, int event_size, float first_timestamp, float duration, double fx, double fy, double cx, double cy, float theta)
{
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    float duration_inv=1.0/duration;
    for (size_t i = thread_grid_idx; i < event_size; i += num_threads_in_grid)
    {
        double event_theta = theta * (et[i] - first_timestamp) * duration_inv;
        double math_s = 1.0/(fx - event_theta * (exy[i * 2] - cx));
        int new_x = (exy[i * 2] * (fx - cx * event_theta) + event_theta * (fx * fx + cx * cx)) * math_s + 200;
        int new_y = ((double)exy[i * 2 + 1] * fx - cy * event_theta * (double)exy[i * 2] + cx * cy * event_theta) * math_s;
        if (new_x >= 0 && new_x < 640 && new_y >= 0 && new_y < 480)
        {
            // atomicAdd(mat_out+(new_x+new_y*640)*sizeof(int32_t),(int32_t)1);
            atomicAdd(mat_out + (new_x + new_y * 640), (int32_t)1);
        }
        // mat_out[new_x+new_y*640]++;
    }
}
__global__ void accumulate_(uint16_t *exy, int32_t *mat_out, int event_size)
{
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t i = thread_grid_idx; i < event_size; i += num_threads_in_grid)
    {
        
        atomicAdd(mat_out + (exy[i * 2] + exy[i * 2+1] * 640), (int32_t)1);
        
        // mat_out[new_x+new_y*640]++;
    }
}

void motionCompensate(uint16_t *exy, float *et, int32_t *mat_out, int event_size, float first_timestamp, float duration, double fx, double fy, double cx, double cy, float theta)
{
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    motionCompensate_<<<blocks_per_sm * num_sm, threads_per_block>>>(exy, et, mat_out, event_size, first_timestamp, duration, fx, fy, cx, cy, theta);
}

void accumulate(uint16_t *exy, int32_t *mat_out, int event_size)
{
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    accumulate_<<<blocks_per_sm * num_sm, threads_per_block>>>(exy, mat_out, event_size);
}
__global__ void rescale_(int *mat, uint8_t *mat_out, float scale)
{
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    // for(size_t i = thread_grid_idx; i<1024*768; i += num_threads_in_grid) {
    for (size_t i = thread_grid_idx; i < 640 * 480*2; i += num_threads_in_grid)
    {
        // mat_out[i] = clamp((float)(mat[i]*scale),(float)0,(float)255);
        mat_out[i] = (uint8_t)(min(max((float)(mat[i] * scale), (float)0), (float)255));
    }
}

void rescale(int *mat, uint8_t *mat_out, float scale)
{
    const int num_sm = 8; // Jetson Orin NX
    const int blocks_per_sm = 4;
    const int threads_per_block = 128;
    rescale_<<<blocks_per_sm * num_sm, threads_per_block>>>(mat, mat_out, scale);
}