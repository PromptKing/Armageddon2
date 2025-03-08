#include "MotionEstimationKernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath> // For fabsf
#include <device_launch_parameters.h>


// Function to invoke the CUDA kernel for motion estimation
bool RunMotionEstimation(
    const float* d_prevFrame,
    const float* d_currFrame,
    float* d_motionVectors,
    int width,
    int height
) {
    // Define block and grid sizes
    dim3 blockSize(16, 16); // Each block contains 16x16 threads
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x, // Number of blocks in the x direction
        (height + blockSize.y - 1) / blockSize.y // Number of blocks in the y direction
    );

    // Synchronize the device to check for kernel errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}
