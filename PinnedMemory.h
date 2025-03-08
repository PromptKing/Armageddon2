#ifndef PINNED_MEMORY_H
#define PINNED_MEMORY_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <cuda.h>

#include <device_launch_parameters.h>

#include <cuda_runtime_api.h> // Updated include

#include <driver_types.h>


// A helper class to manage pinned memory operations
class PinnedMemory {
public:
    // Allocate pinned memory for the given size
    template <typename T>
    static T* Allocate(size_t size) {
        T* memory = nullptr;
        cudaError_t err = cudaMallocHost((void**)&memory, size * sizeof(T));
        if (err != cudaSuccess) {
            std::cerr << "Error: Failed to allocate pinned memory. CUDA error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Failed to allocate pinned memory.");
        }
        return memory;
    }

    // Free pinned memory
    template <typename T>
    static void Free(T* memory) {
        cudaError_t err = cudaFreeHost(memory);
        if (err != cudaSuccess) {
            std::cerr << "Error: Failed to free pinned memory. CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Copy data to the device
    template <typename T>
    static void CopyToDevice(const T* hostMemory, T* deviceMemory, size_t size) {
        cudaError_t err = cudaMemcpy(deviceMemory, hostMemory, size * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Error: Failed to copy data to device. CUDA error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Failed to copy data to device.");
        }
    }

    // Copy data from the device
    template <typename T>
    static void CopyFromDevice(const T* deviceMemory, T* hostMemory, size_t size) {
        cudaError_t err = cudaMemcpy(hostMemory, deviceMemory, size * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Error: Failed to copy data from device. CUDA error: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Failed to copy data from device.");
        }
    }
};

#endif // PINNED_MEMORY_H
