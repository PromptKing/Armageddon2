#include "MemoryBridge.h"  // Include the MemoryBridge header

#include "Metadata.h"      // Include the Metadata structure definition
#include <cuda_runtime.h>  // Include CUDA runtime API for GPU operations
#include <chrono>          // For time-related operations
#include <iostream>        // For console output
#include <cstring>         // For memcpy
#include <mutex>           // For thread-safe access

MemoryBridge::MemoryBridge(size_t bufferSize)
    : bufferSize(bufferSize), gpuBuffer(nullptr) {
}

MemoryBridge::~MemoryBridge() {
    ReleaseResources();
}

bool MemoryBridge::Initialize() {
    // Allocate GPU memory
    cudaError_t err = cudaMalloc(&gpuBuffer, bufferSize);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to allocate GPU memory. CUDA error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    std::cout << "GPU memory allocated successfully. Size: " << bufferSize << " bytes\n";
    return true;
}

void MemoryBridge::ReleaseResources() {
    if (gpuBuffer) {
        cudaFree(gpuBuffer);
        gpuBuffer = nullptr;
        std::cout << "GPU memory released.\n";
    }
}

bool MemoryBridge::CopyDataToGPU(const std::vector<float>& data) {
    if (!gpuBuffer) {
        std::cerr << "Error: GPU buffer not initialized.\n";
        return false;
    }

    if (data.size() * sizeof(float) > bufferSize) {
        std::cerr << "Error: Data size exceeds buffer size.\n";
        return false;
    }

    cudaError_t err = cudaMemcpy(gpuBuffer, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to copy data to GPU. CUDA error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    std::cout << "Data successfully copied to GPU.\n";
    return true;
}

bool MemoryBridge::RetrieveDataFromGPU(std::vector<float>& data) {
    if (!gpuBuffer) {
        std::cerr << "Error: GPU buffer not initialized.\n";
        return false;
    }

    data.resize(bufferSize / sizeof(float));
    cudaError_t err = cudaMemcpy(data.data(), gpuBuffer, bufferSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error: Failed to retrieve data from GPU. CUDA error: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    std::cout << "Data successfully retrieved from GPU.\n";
    return true;
}

void MemoryBridge::CacheMetadata(int threadID, const Metadata& metadata) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    metadataCache[threadID] = metadata;
    std::cout << "Metadata cached for thread ID: " << threadID << "\n";
}

bool MemoryBridge::RetrieveMetadata(int threadID, Metadata& metadata) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto it = metadataCache.find(threadID);
    if (it != metadataCache.end()) {
        metadata = it->second;
        return true;
    }

    std::cerr << "Error: Metadata not found for thread ID: " << threadID << "\n";
    return false;
}

double MemoryBridge::GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}
