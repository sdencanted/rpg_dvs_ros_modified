#include "dvxplorer_mc_ceres/motion_compensation.cuh"

#include <cmath>
#include <algorithm> //for std::max
#include <cstdio>
#include <vector>
#include <iostream>
#include "dvxplorer_mc_ceres/utils.h"

#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>

#include <jetson-utils/cudaMappedMemory.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>

#define FULL_MASK 0xffffffff

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

__global__ void g_motionCompensate(float fx, float fy, float cx, float cy, int height, int width, int num_events, const float *x_unprojected, const float *y_unprojected, float *x_prime, float *y_prime, float *t, const float rotation_x, const float rotation_y, const float rotation_z, int x_offset, int y_offset)
{
    // size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    int total_cx = cx - x_offset;
    int total_cy = cy - y_offset;
    // if (i < num_events)
    for (size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x); i < num_events; i += num_threads_in_grid)
    {
        // calculate theta x,y,z
        float theta_x_t = rotation_x * t[i];
        float theta_y_t = rotation_y * t[i];
        float theta_z_t = rotation_z * t[i];

        // calculate x/y/z_rotated
        float z_rotated_inv = 1 / (-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
        float x_rotated_norm = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv;
        float y_rotated_norm = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv;

        // calculate x_prime and y_prime
        x_prime[i] = fx * x_rotated_norm + total_cx;
        y_prime[i] = fy * y_rotated_norm + total_cy;
    }
}

__global__ void g_motionCompensateAndFillImageGaussian5(float fx, float fy, float cx, float cy, int height, int width, int num_events, const float *x_unprojected, const float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int x_offset, int y_offset)
{

    float image_sum = 0;
    float image_sum_del_theta_x = 0;
    float image_sum_del_theta_y = 0;
    float image_sum_del_theta_z = 0;
    float *image_del_x = image + height * width;
    float *image_del_y = image + height * width * 2;
    float *image_del_z = image + height * width * 3;
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    float t_mid = (t[num_events - 1] + t[0]) / 2;
    int total_cx = cx - x_offset;
    int total_cy = cy - y_offset;
    for (size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x); i < num_events; i += num_threads_in_grid)
    {
        float t_norm = t[i] - t_mid;
        // calculate theta x,y,z
        float theta_x_t = rotation_x * t_norm;
        float theta_y_t = rotation_y * t_norm;
        float theta_z_t = rotation_z * t_norm;

        // calculate x/y/z_rotated
        float z_rotated_inv = 1 / (-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
        float x_rotated_norm = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv;
        float y_rotated_norm = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv;

        // calculate x_prime and y_prime
        x_prime[i] = fx * x_rotated_norm + total_cx;
        y_prime[i] = fy * y_rotated_norm + total_cy;
        // populate image
        int x_round = round(x_prime[i]);
        int y_round = round(y_prime[i]);
        float gaussian;

        if (x_round >= 1 && x_round <= width && y_round >= 1 && y_round <= height)
        {
            float fx_div_z_rotated_ti = fx * z_rotated_inv * t_norm;
            float fy_div_z_rotated_ti = fy * z_rotated_inv * t_norm;
            float del_x_del_theta_y = fx_div_z_rotated_ti * (1 + x_unprojected[i] * x_rotated_norm);
            float del_x_del_theta_z = -fx_div_z_rotated_ti * y_unprojected[i];
            float del_x_del_theta_x = del_x_del_theta_z * x_rotated_norm;
            float del_y_del_theta_x = fy_div_z_rotated_ti * (-1 - y_unprojected[i] * y_rotated_norm);
            float del_y_del_theta_z = fy_div_z_rotated_ti * x_unprojected[i];
            float del_y_del_theta_y = del_y_del_theta_z * y_rotated_norm;

            for (int row = max(1, y_round - 2); row < min(height, y_round + 3); row++)
            {
                for (int col = max(1, x_round - 2); col < min(width, x_round + 3); col++)
                {
                    float x_diff = col - x_prime[i];
                    float y_diff = row - y_prime[i];
                    gaussian = exp((-x_diff * x_diff - y_diff * y_diff) / 2);
                    int idx = (row - 1) * (width) + col - 1;
                    atomicAdd(&image[idx], gaussian);
                    image_sum += gaussian;
                    float del_x = gaussian * (x_diff * del_x_del_theta_x + y_diff * del_y_del_theta_x);
                    atomicAdd(&image_del_x[idx], del_x);
                    image_sum_del_theta_x += del_x;
                    float del_y = gaussian * (x_diff * del_x_del_theta_y + y_diff * del_y_del_theta_y);
                    atomicAdd(&image_del_y[idx], del_y);
                    image_sum_del_theta_y += del_y;
                    float del_z = gaussian * (x_diff * del_x_del_theta_z + y_diff * del_y_del_theta_z);
                    atomicAdd(&image_del_z[idx], del_z);
                    image_sum_del_theta_z += del_z;
                }
            }
        }
    }
    float *sdata = SharedMemory<float>();
    uint16_t tid = threadIdx.x;

    // do reduction in shared mem

    // sum up to 128 elements

    float temp_sum;
    // image_sum
    sdata[tid] = image_sum;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum = image_sum + sdata[tid + 256];
    __syncthreads();
    // store contrast in 0 to 127
    if (tid < 128)
        temp_sum = image_sum + sdata[tid + 128];
    __syncthreads();
    // image_sum_del_theta_x
    sdata[tid] = image_sum_del_theta_x;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_x = image_sum_del_theta_x + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_sum_del_theta_x = image_sum_del_theta_x + sdata[tid + 128];
    __syncthreads();
    // store x in 128 to 255
    if (tid >= 128 && tid < 256)
    {
        temp_sum = sdata[tid - 128];
    }
    __syncthreads();
    // image_sum_del_theta_y
    sdata[tid] = image_sum_del_theta_y;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_y = image_sum_del_theta_y + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_sum_del_theta_y = image_sum_del_theta_y + sdata[tid + 128];
    __syncthreads();
    // store y in 256 to 383
    if (tid >= 256 && tid < 384)
    {
        temp_sum = sdata[tid - 256];
    }
    __syncthreads();
    // image_sum_del_theta_z
    sdata[tid] = image_sum_del_theta_z;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_z = image_sum_del_theta_z + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
    {
        sdata[tid] = image_sum_del_theta_z = image_sum_del_theta_z + sdata[tid + 128];
    }
    __syncthreads();
    // store z in 384 to 512
    if (tid >= 384)
    {
        temp_sum = sdata[tid - 384];
    }
    // dump partial sums inside again
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid & 0x7F) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid & 0x7F) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    __syncthreads();

    if (tid == 0)
    {
        // image_sum
        contrast_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 128)
    {
        // image_sum_del_theta_x
        contrast_del_x_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 256)
    {
        // image_sum_del_theta_y
        contrast_del_y_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 384)
    {
        // image_sum_del_theta_x
        contrast_del_z_block_sum[blockIdx.x] = temp_sum;
    }
}

__global__ void g_motionCompensateAndFillImageBilinear(float fx, float fy, float cx, float cy, int height, int width, int num_events, const float *x_unprojected, const float *y_unprojected, float *x_prime, float *y_prime, float *t, float *image, const float rotation_x, const float rotation_y, const float rotation_z, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int x_offset, int y_offset)
{

    float image_sum = 0;
    float image_sum_del_theta_x = 0;
    float image_sum_del_theta_y = 0;
    float image_sum_del_theta_z = 0;
    float *image_del_x = image + height * width;
    float *image_del_y = image + height * width * 2;
    float *image_del_z = image + height * width * 3;
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    float t_mid = (t[num_events - 1] + t[0]) / 2;
    int total_cx = cx - x_offset;
    int total_cy = cy - y_offset;
    for (size_t i = size_t(blockIdx.x * blockDim.x + threadIdx.x); i < num_events; i += num_threads_in_grid)
    {
        float t_norm = t[i] - t_mid;
        // calculate theta x,y,z
        float theta_x_t = rotation_x * t_norm;
        float theta_y_t = rotation_y * t_norm;
        float theta_z_t = rotation_z * t_norm;

        // calculate x/y/z_rotated
        float z_rotated_inv = 1 / (-theta_y_t * x_unprojected[i] + theta_x_t * y_unprojected[i] + 1);
        float x_rotated_norm = (x_unprojected[i] - theta_z_t * y_unprojected[i] + theta_y_t) * z_rotated_inv;
        float y_rotated_norm = (theta_z_t * x_unprojected[i] + y_unprojected[i] - theta_x_t) * z_rotated_inv;

        // calculate x_prime and y_prime
        x_prime[i] = fx * x_rotated_norm + total_cx;
        y_prime[i] = fy * y_rotated_norm + total_cy;
        // populate image

        // Bilinear
        int x_trunc = int(x_prime[i]);
        int y_trunc = int(y_prime[i]);
        if (x_trunc >= 1 && x_trunc <= width - 2 && y_trunc >= 1 && y_trunc <= height - 2)
        {

            // int idx1 = x_trunc - 1 + (y_trunc - 1) * width;
            // int idx2 = idx1 + 1;
            // int idx3 = idx1 + width;
            // int idx4 = idx3 + 1;

            int idx4 = x_trunc + y_trunc * width;
            int idx3 = idx4 - 1;
            int idx2 = idx4 - width;
            int idx1 = idx2 - 1;
            float x_diff = x_prime[i] - x_trunc;
            float y_diff = y_prime[i] - y_trunc;
            float del_x_del_theta_x, del_x_del_theta_y, del_x_del_theta_z, del_y_del_theta_x, del_y_del_theta_y, del_y_del_theta_z;
            float fx_div_z_rotated_ti = fx * z_rotated_inv * t[i];
            float fy_div_z_rotated_ti = fy * z_rotated_inv * t[i];
            del_x_del_theta_y = fx_div_z_rotated_ti * (1 + x_unprojected[i] * x_rotated_norm);
            del_x_del_theta_z = fx_div_z_rotated_ti * -y_unprojected[i];
            del_x_del_theta_x = del_x_del_theta_z * x_rotated_norm;
            del_y_del_theta_x = fy_div_z_rotated_ti * (-1 - y_unprojected[i] * y_rotated_norm);
            del_y_del_theta_z = fy_div_z_rotated_ti * x_unprojected[i];
            del_y_del_theta_y = del_y_del_theta_z * y_rotated_norm;
            // float d1x = -(1 - y_diff);
            // float d1y = -(1 - x_diff);
            float d2x = 1 - y_diff;
            float d2y = -x_diff;
            float d3x = -y_diff;
            float d3y = 1 - x_diff;
            float d4x = y_diff;
            float d4y = x_diff;

            float d1x = -d2x;
            float d1y = -d3y;

            float im1 = d3y * d2x;
            float im2 = d4y * d2x;
            float im3 = d3y * y_diff;
            float im4 = (x_diff) * (y_diff);
            image_sum += im1 + im2 + im3 + im4;
            atomicAdd(&image[idx1], im1);
            atomicAdd(&image[idx2], im2);
            atomicAdd(&image[idx3], im3);
            atomicAdd(&image[idx4], im4);
            float dx1 = d1x * del_x_del_theta_x + d1y * del_y_del_theta_x;
            float dx2 = d2x * del_x_del_theta_x + d2y * del_y_del_theta_x;
            float dx3 = d3x * del_x_del_theta_x + d3y * del_y_del_theta_x;
            float dx4 = d4x * del_x_del_theta_x + d4y * del_y_del_theta_x;
            image_sum_del_theta_x += dx1 + dx2 + dx3 + dx4;

            atomicAdd(&image_del_x[idx1], dx1);
            atomicAdd(&image_del_x[idx2], dx2);
            atomicAdd(&image_del_x[idx3], dx3);
            atomicAdd(&image_del_x[idx4], dx4);
            float dy1 = d1x * del_x_del_theta_y + d1y * del_y_del_theta_y;
            float dy2 = d2x * del_x_del_theta_y + d2y * del_y_del_theta_y;
            float dy3 = d3x * del_x_del_theta_y + d3y * del_y_del_theta_y;
            float dy4 = d4x * del_x_del_theta_y + d4y * del_y_del_theta_y;
            image_sum_del_theta_y += dy1 + dy2 + dy3 + dy4;
            atomicAdd(&image_del_y[idx1], dy1);
            atomicAdd(&image_del_y[idx2], dy2);
            atomicAdd(&image_del_y[idx3], dy3);
            atomicAdd(&image_del_y[idx4], dy4);
            float dz1 = d1x * del_x_del_theta_z + d1y * del_y_del_theta_z;
            float dz2 = d2x * del_x_del_theta_z + d2y * del_y_del_theta_z;
            float dz3 = d3x * del_x_del_theta_z + d3y * del_y_del_theta_z;
            float dz4 = d4x * del_x_del_theta_z + d4y * del_y_del_theta_z;
            image_sum_del_theta_z += dz1 + dz2 + dz3 + dz4;
            atomicAdd(&image_del_z[idx1], dz1);
            atomicAdd(&image_del_z[idx2], dz2);
            atomicAdd(&image_del_z[idx3], dz3);
            atomicAdd(&image_del_z[idx4], dz4);
        }
    }
    float *sdata = SharedMemory<float>();
    uint16_t tid = threadIdx.x;

    // do reduction in shared mem

    // sum up to 128 elements

    float temp_sum;
    // image_sum
    sdata[tid] = image_sum;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum = image_sum + sdata[tid + 256];
    __syncthreads();
    // store contrast in 0 to 127
    if (tid < 128)
        temp_sum = image_sum + sdata[tid + 128];
    __syncthreads();
    // image_sum_del_theta_x
    sdata[tid] = image_sum_del_theta_x;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_x = image_sum_del_theta_x + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_sum_del_theta_x = image_sum_del_theta_x + sdata[tid + 128];
    __syncthreads();
    // store x in 128 to 255
    if (tid >= 128 && tid < 256)
    {
        temp_sum = sdata[tid - 128];
    }
    __syncthreads();
    // image_sum_del_theta_y
    sdata[tid] = image_sum_del_theta_y;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_y = image_sum_del_theta_y + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_sum_del_theta_y = image_sum_del_theta_y + sdata[tid + 128];
    __syncthreads();
    // store y in 256 to 383
    if (tid >= 256 && tid < 384)
    {
        temp_sum = sdata[tid - 256];
    }
    __syncthreads();
    // image_sum_del_theta_z
    sdata[tid] = image_sum_del_theta_z;
    __syncthreads();
    if (tid < 256)
        sdata[tid] = image_sum_del_theta_z = image_sum_del_theta_z + sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
    {
        sdata[tid] = image_sum_del_theta_z = image_sum_del_theta_z + sdata[tid + 128];
    }
    __syncthreads();
    // store z in 384 to 512
    if (tid >= 384)
    {
        temp_sum = sdata[tid - 384];
    }
    // dump partial sums inside again
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid & 0x7F) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid & 0x7F) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    __syncthreads();

    if (tid == 0)
    {
        // image_sum
        contrast_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 128)
    {
        // image_sum_del_theta_x
        contrast_del_x_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 256)
    {
        // image_sum_del_theta_y
        contrast_del_y_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 384)
    {
        // image_sum_del_theta_x
        contrast_del_z_block_sum[blockIdx.x] = temp_sum;
    }
}

__global__ void g_fillImageKroneckerNoJacobians(int height, int width, int num_events, float *x_prime, float *y_prime, int *image)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t i = thread_grid_idx; i < num_events; i += num_threads_in_grid)
    {
        // populate image
        // check if coordinates are 3 pixels in of the boundary
        int x_round = round(x_prime[i]);
        int y_round = round(y_prime[i]);
        if (x_round >= 1 && x_round <= width && y_round >= 1 && y_round <= height)
        {
            int idx = (y_round - 1) * width + x_round - 1;
            atomicAdd(&image[idx], 1);
        }
    }
}
__global__ void g_fillImageBilinearNoJacobians(int height, int width, int num_events, float *x_prime, float *y_prime, float *image)
{
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);
    for (size_t i = thread_grid_idx; i < num_events; i += num_threads_in_grid)
    {
        // Bilinear
        int x_trunc = int(x_prime[i]);
        int y_trunc = int(y_prime[i]);
        if (x_trunc >= 1 && x_trunc <= width - 2 && y_trunc >= 1 && y_trunc <= height - 2)
        {

            int idx4 = x_trunc + y_trunc * width;
            int idx3 = idx4 - 1;
            int idx2 = idx4 - width;
            int idx1 = idx2 - 1;
            float x_diff = x_prime[i] - x_trunc;
            float y_diff = y_prime[i] - y_trunc;
            float d2x = 1 - y_diff;
            float d3y = 1 - x_diff;
            float d4y = x_diff;

            float im1 = d3y * d2x;
            float im2 = d4y * d2x;
            float im3 = d3y * y_diff;
            float im4 = (x_diff) * (y_diff);
            atomicAdd(&image[idx1], im1);
            atomicAdd(&image[idx2], im2);
            atomicAdd(&image[idx3], im3);
            atomicAdd(&image[idx4], im4);
        }
    }
}
__device__ volatile float mean_volatile[4] = {0};

template <int block_size>
__global__ void g_calculateAndReduceContrastAndJacobians(float *image, int num_elements, float *means, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    float *image_del_x = image + num_elements;
    float *image_del_y = image + num_elements * 2;
    float *image_del_z = image + num_elements * 3;
    // START COPY
    float *sdata = SharedMemory<float>();
    float temp_sum = 0;
    uint16_t tid = threadIdx.x;
    // 85 partial sums to go
    // dump partial sums inside again
    // if (tid < prev_gridsize)
    for (int i = tid; i < prev_gridsize; i += blockDim.x)
    {

        if (blockIdx.x == 0)
        {
            temp_sum += contrast_block_sum[i];
        }
        else if (blockIdx.x == 1)
        {
            temp_sum += contrast_del_x_block_sum[i];
        }
        else if (blockIdx.x == 2)
        {
            temp_sum += contrast_del_y_block_sum[i];
        }
        else if (blockIdx.x == 3)
        {
            temp_sum += contrast_del_z_block_sum[i];
        }
    }
    sdata[tid] = temp_sum;
    __syncthreads();
    if (block_size > 256)
    {
        // 512 threads per block
        if ((tid) < 256)
        {
            sdata[tid] = temp_sum = temp_sum + sdata[tid + 128];
        }
        __syncthreads();
    }
    if (block_size > 128)
    {
        if ((tid) < 128)
        {
            sdata[tid] = temp_sum = temp_sum + sdata[tid + 128];
        }
        __syncthreads();
    }
    if (block_size > 64)
    {
        if ((tid) < 64)
        {
            sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
        }
        __syncthreads();
    }
    if ((tid) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    if (tid == 0)
    {
        // write means
        if (blockIdx.x < 4)
        {
            mean_volatile[blockIdx.x] = temp_sum / num_elements;
        }
    }
    // __syncthreads();

    // END COPY
    float image_contrast = 0;
    float image_contrast_del_theta_x = 0;
    float image_contrast_del_theta_y = 0;
    float image_contrast_del_theta_z = 0;
    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t idx = thread_grid_idx;
    cooperative_groups::sync(grid);
    float mean = mean_volatile[0];
    float xmean = mean_volatile[1];
    float ymean = mean_volatile[2];
    float zmean = mean_volatile[3];

    while (idx < num_elements)
    {
        float image_norm = image[idx] - mean;
        float image_norm_x = image_del_x[idx] - xmean;
        float image_norm_y = image_del_y[idx] - ymean;
        float image_norm_z = image_del_z[idx] - zmean;
        image_contrast += image_norm * image_norm;
        image_contrast_del_theta_x += image_norm_x * image_norm;
        image_contrast_del_theta_y += image_norm_y * image_norm;
        image_contrast_del_theta_z += image_norm_z * image_norm;
        // imagedebug[idx] = image_norm * image_norm;
        // imagedebug[idx + num_elements] = image_norm_x * image_norm;
        // imagedebug[idx + num_elements * 2] = image_norm_y * image_norm;
        // imagedebug[idx + num_elements * 3] = image_norm_z * image_norm;
        idx += blockDim.x * gridDim.x;
    }
    // BEGIN DEBUG
    // sdata[tid] = 0;
    // if(blockIdx.x==0&&threadIdx.x==0){
    //     for(int idx_special=0;idx_special<num_elements;idx_special++)

    //     {
    //         float image_norm = image[idx_special] - mean;
    //         float image_norm_x = image_del_x[idx_special] - xmean;
    //         float image_norm_y = image_del_y[idx_special] - ymean;
    //         float image_norm_z = image_del_z[idx_special] - zmean;
    //         // image_contrast += image_norm * image_norm;
    //         // image_contrast_del_theta_x += image_norm_x * image_norm;
    //         // image_contrast_del_theta_y += image_norm_y * image_norm;
    //         // image_contrast_del_theta_z += image_norm_z * image_norm;
    //         // imagedebug[idx_special] = image_norm * image_norm;
    //         // imagedebug[idx_special + num_elements] = image_norm_x * image_norm;
    //         // imagedebug[idx_special + num_elements * 2] = image_norm_y * image_norm;
    //         // imagedebug[idx_special + num_elements * 3] = image_norm_z * image_norm;

    //         atomicAdd(&sdata[0], image_norm * image_norm);
    //         atomicAdd(&sdata[1], image_norm_x * image_norm);
    //         atomicAdd(&sdata[2], image_norm_y * image_norm);
    //         atomicAdd(&sdata[3], image_norm_z * image_norm);

    //     }

    //     contrast_block_sum[blockIdx.x] = sdata[0];
    //     contrast_del_x_block_sum[blockIdx.x] = sdata[1];
    //     contrast_del_y_block_sum[blockIdx.x] = sdata[2];
    //     contrast_del_z_block_sum[blockIdx.x] = sdata[3];
    //     // contrast_block_sum[blockIdx.x] = image_contrast;
    //     // contrast_del_x_block_sum[blockIdx.x] = image_contrast_del_theta_x;
    //     // contrast_del_y_block_sum[blockIdx.x] = image_contrast_del_theta_y;
    //     // contrast_del_z_block_sum[blockIdx.x] = image_contrast_del_theta_z;
    // }
    // else{

    //     contrast_block_sum[blockIdx.x] =0;
    //     contrast_del_x_block_sum[blockIdx.x] =0;
    //     contrast_del_y_block_sum[blockIdx.x] =0;
    //     contrast_del_z_block_sum[blockIdx.x] = 0;
    // }
    // return;
    // END DEBUG

    // do reduction in shared mem

    // sum up to 128 elements

    // float temp_sum;
    // image_contrast
    sdata[tid] = image_contrast;
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast = image_contrast + sdata[tid + 128];
    __syncthreads();
    // store contrast in 0 to 63
    if (tid < 64)
        temp_sum = image_contrast + sdata[tid + 64];
    __syncthreads();
    // image_contrast_del_theta_x
    sdata[tid] = image_contrast_del_theta_x;
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast_del_theta_x = image_contrast_del_theta_x + sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] = image_contrast_del_theta_x = image_contrast_del_theta_x + sdata[tid + 64];
    __syncthreads();
    // store x in 64 to 127
    if (tid >= 64 && tid < 128)
    {
        temp_sum = sdata[tid - 64];
    }
    __syncthreads();
    // image_contrast_del_theta_y
    sdata[tid] = image_contrast_del_theta_y;
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast_del_theta_y = image_contrast_del_theta_y + sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] = image_contrast_del_theta_y = image_contrast_del_theta_y + sdata[tid + 64];
    __syncthreads();
    // store y in 128 to 191
    if (tid >= 128 && tid < 192)
    {
        temp_sum = sdata[tid - 128];
    }
    __syncthreads();
    // image_contrast_del_theta_z
    sdata[tid] = image_contrast_del_theta_z;
    __syncthreads();
    if (tid < 128)
        sdata[tid] = image_contrast_del_theta_z = image_contrast_del_theta_z + sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
    {
        sdata[tid] = image_contrast_del_theta_z = image_contrast_del_theta_z + sdata[tid + 64];
    }
    __syncthreads();
    // store z in 192 to 255
    if (tid >= 192)
    {
        temp_sum = sdata[tid - 192];
    }
    __syncthreads();
    // dump partial sums inside again
    sdata[tid] = temp_sum;
    __syncthreads();
    if ((tid & 0x3F) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }

    if (tid == 0)
    {
        // image_contrast
        contrast_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 64)
    {
        // image_contrast_del_theta_x
        contrast_del_x_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 128)
    {
        // image_contrast_del_theta_y
        contrast_del_y_block_sum[blockIdx.x] = temp_sum;
    }
    else if (tid == 192)
    {
        // image_contrast_del_theta_x
        contrast_del_z_block_sum[blockIdx.x] = temp_sum;
    }
}
template __global__ void g_calculateAndReduceContrastAndJacobians<512>(float *image, int num_elements, float *means, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);
template __global__ void g_calculateAndReduceContrastAndJacobians<256>(float *image, int num_elements, float *means, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);
template __global__ void g_calculateAndReduceContrastAndJacobians<128>(float *image, int num_elements, float *means, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);
template __global__ void g_calculateAndReduceContrastAndJacobians<64>(float *image, int num_elements, float *means, float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);
// 4 blocks x threads
template <int prev_block_size>
__global__ void g_reduceContrastAndJacobiansPt2(float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize)
{
    float *sdata = SharedMemory<float>();

    float temp_sum = 0;
    uint16_t tid = threadIdx.x;
    // 85 partial sums to go
    // dump partial sums inside again
    if (tid < prev_gridsize)
    {

        if (blockIdx.x == 0)
        {
            temp_sum = temp_sum = contrast_block_sum[tid];
        }
        else if (blockIdx.x == 1)
        {
            temp_sum = contrast_del_x_block_sum[tid];
        }
        else if (blockIdx.x == 2)
        {
            temp_sum = contrast_del_y_block_sum[tid];
        }
        else if (blockIdx.x == 3)
        {
            temp_sum = contrast_del_z_block_sum[tid];
        }
    }
    sdata[tid] = temp_sum;
    __syncthreads();

    if (prev_block_size > 256 && (tid) < 256)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 256];
    }
    __syncthreads();
    if (prev_block_size > 128 && (tid) < 128)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 128];
    }
    __syncthreads();
    if (prev_block_size > 64 && (tid) < 64)
    {
        sdata[tid] = temp_sum = temp_sum + sdata[tid + 64];
    }
    __syncthreads();
    if ((tid) < 32)
    {
        // warps of 32 threads are always in sync, no need to sync after this
        temp_sum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (uint8_t offset = 32 / 2; offset > 0; offset = offset >> 1)
        {
            temp_sum += __shfl_down_sync(FULL_MASK, temp_sum, offset);
        }
    }
    if (tid == 0)
    {
        if (blockIdx.x == 0)
        {
            contrast_block_sum[0] = temp_sum;
        }
        else if (blockIdx.x == 1)
        {
            contrast_block_sum[1] = temp_sum;
        }
        else if (blockIdx.x == 2)
        {
            contrast_block_sum[2] = temp_sum;
        }
        else
        {
            contrast_block_sum[3] = temp_sum;
        }
    }
}
template __global__ void g_reduceContrastAndJacobiansPt2<512>(float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);
template __global__ void g_reduceContrastAndJacobiansPt2<256>(float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);
template __global__ void g_reduceContrastAndJacobiansPt2<128>(float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);
template __global__ void g_reduceContrastAndJacobiansPt2<64>(float *contrast_block_sum, float *contrast_del_x_block_sum, float *contrast_del_y_block_sum, float *contrast_del_z_block_sum, int prev_gridsize);

__global__ void g_rescaleIntensityInt(int *image, uint8_t *output_image, int maximum, int numel)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t idx = thread_grid_idx; idx < numel; idx += num_threads_in_grid)
    {
        output_image[idx] = (uint8_t)min(255, max(0, (int)(255 * image[idx] / (maximum / 2))));
    }
}
__global__ void g_rescaleIntensityFloat(float *image, uint8_t *output_image, float maximum, int numel)
{

    size_t thread_grid_idx = size_t(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads_in_grid = size_t(blockDim.x * gridDim.x);

    for (size_t idx = thread_grid_idx; idx < numel; idx += num_threads_in_grid)
    {
        output_image[idx] = (uint8_t)min(255, max(0, (int)(255.0 * image[idx] / (maximum / 2))));
    }
}
