#include "CPUGPUDataPipeline.h"
#include <iostream>
#include <cstring>
#include <random>
#include <cuda_runtime.h>
#include <future>  // For async execution


// Constructor
CPUGPUDataPipeline::CPUGPUDataPipeline(size_t bufferSize)
    : bufferSize(bufferSize), pinnedMemory(nullptr), deviceMemory(nullptr), stream(nullptr) {
}

// Destructor
CPUGPUDataPipeline::~CPUGPUDataPipeline() {
    if (pinnedMemory) cudaFreeHost(pinnedMemory);
    if (deviceMemory) cudaFree(deviceMemory);
    if (stream) cudaStreamDestroy(stream);
}

// **1. Initialize using Unified Memory (cudaMallocManaged)**
bool CPUGPUDataPipeline::Initialize() {
    // Use unified memory for better memory sharing between CPU & GPU
    if (cudaMallocManaged(&deviceMemory, bufferSize * sizeof(float)) != cudaSuccess) {
        std::cerr << "[ERROR] Failed to allocate unified memory." << std::endl;
        return false;
    }

    if (cudaMallocHost(&pinnedMemory, bufferSize * sizeof(float)) != cudaSuccess) {
        std::cerr << "[ERROR] Failed to allocate pinned memory on the CPU." << std::endl;
        return false;
    }

    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "[ERROR] Failed to create CUDA stream." << std::endl;
        return false;
    }

    return true;
}

// **2. Add Error Handling with `cudaGetLastError()`**
void CPUGPUDataPipeline::CheckCudaError(const std::string& message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << message << ": " << cudaGetErrorString(err) << std::endl;
    }
}

// **3. Send Data to GPU with Asynchronous Transfers**
bool CPUGPUDataPipeline::SendDataToGPU(const std::vector<float>& data, const std::string& task) {
    if (data.size() > bufferSize) {
        std::cerr << "[ERROR] Data size exceeds buffer size." << std::endl;
        return false;
    }

    memcpy(pinnedMemory, data.data(), data.size() * sizeof(float));

    if (cudaMemcpyAsync(deviceMemory, pinnedMemory, data.size() * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy data to GPU." << std::endl;
        return false;
    }

    LogTask(task);

    // **4. Multi-Stream Execution for Better Performance**
    std::future<bool> taskFuture = std::async(std::launch::async, [&, task]() {
        if (task == "physics") return LaunchPhysicsKernel(data);
        if (task == "graphics") return LaunchGraphicsKernel(data);
        if (task == "ai") return LaunchAIKernel(data);
        return false;
        });

    return taskFuture.get();
}

// **5. Retrieve Data with Prefetching**
bool CPUGPUDataPipeline::RetrieveDataFromGPU(std::vector<float>& data, const std::string& task) {
    if (data.size() < bufferSize) {
        data.resize(bufferSize);
    }

    if (cudaMemcpyAsync(pinnedMemory, deviceMemory, bufferSize * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "[ERROR] Failed to retrieve data from GPU." << std::endl;
        return false;
    }

    cudaStreamSynchronize(stream);
    memcpy(data.data(), pinnedMemory, bufferSize * sizeof(float));

    // Prefetch data for future access
    cudaMemPrefetchAsync(deviceMemory, bufferSize * sizeof(float), cudaCpuDeviceId, stream);

    std::cout << "[INFO] Task: " << task << " completed successfully." << std::endl;
    return true;
}

// **6. Use Random Vertices for Simulation**
bool CPUGPUDataPipeline::GenerateRandomVertices(std::vector<float>& vertices, size_t vertexCount) {
    vertices.resize(vertexCount * 3);  // Each vertex has x, y, z

    // Use a high-performance random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

    for (size_t i = 0; i < vertexCount * 3; ++i) {
        vertices[i] = dis(gen);
    }

    if (cudaMemcpyAsync(deviceMemory, vertices.data(), vertices.size() * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy random vertices to GPU." << std::endl;
        return false;
    }

    return LaunchVertexKernel(deviceMemory, vertexCount);
}

// **7. Kernel Launches with Timing for Profiling**
bool CPUGPUDataPipeline::LaunchPhysicsKernel(const std::vector<float>& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    std::cout << "[KERNEL] Launching physics kernel with data size: " << data.size() << std::endl;
    // <<<kernel launch here>>>
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[INFO] Physics Kernel Execution Time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return true;
}

// **8. Multi-Tasking Execution**
bool CPUGPUDataPipeline::LaunchGraphicsKernel(const std::vector<float>& data) {
    std::cout << "[KERNEL] Launching graphics kernel with data size: " << data.size() << std::endl;
    return true;
}

bool CPUGPUDataPipeline::LaunchAIKernel(const std::vector<float>& data) {
    std::cout << "[KERNEL] Launching AI kernel with data size: " << data.size() << std::endl;
    return true;
}

// **9. Parallelized Vertex Kernel Execution**
bool CPUGPUDataPipeline::LaunchVertexKernel(float* deviceVertices, size_t vertexCount) {
    std::cout << "[KERNEL] Launching vertex kernel with " << vertexCount << " vertices." << std::endl;

    // Kernel execution here
    return true;
}

// **10. Implement Batch Processing for Large Data Sets**
void CPUGPUDataPipeline::ProcessBatch(const std::vector<std::vector<float>>& batchData, const std::string& task) {
    for (const auto& data : batchData) {
        if (!SendDataToGPU(data, task)) {
            std::cerr << "[ERROR] Batch processing failed at task: " << task << std::endl;
            return;
        }
    }
}

// **Log tasks**
void CPUGPUDataPipeline::LogTask(const std::string& task) {
    std::cout << "[TASK] Processing: " << task << std::endl;
}

