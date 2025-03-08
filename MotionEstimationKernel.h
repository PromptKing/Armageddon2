#include <windows.h>
#include <cuda_runtime.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>
#include <npp.h>

#include <fstream> // For file operations

#include <sstream> // For stringstream


#ifndef MOTION_ESTIMATION_KERNEL_H
#define MOTION_ESTIMATION_KERNEL_H

__global__ void MotionEstimationKernel(
    const float* prevFrame,
    const float* currFrame,
    float* motionVectors,
    int width,
    int height
);

#endif // MOTION_ESTIMATION_KERNEL_H

// CUDA Kernel for motion estimation
__global__ void MotionEstimationKernel(
    const float* prevFrame,
    const float* currFrame,
    float* motionVectors,
    int width,
    int height
);

// Function to invoke the CUDA kernel for motion estimation
bool RunMotionEstimation(
    const float* d_prevFrame,
    const float* d_currFrame,
    float* d_motionVectors,
    int width,
    int height
);

