#ifndef CPUGPU_DATA_PIPELINE_H
#define CPUGPU_DATA_PIPELINE_H

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <future>

class CPUGPUDataPipeline {
public:
    // Constructor
    CPUGPUDataPipeline(size_t bufferSize);

    // Destructor
    ~CPUGPUDataPipeline();

    // Initialize the pipeline resources
    bool Initialize();

    // Send data from CPU to GPU asynchronously
    bool SendDataToGPU(const std::vector<float>& data, const std::string& task);

    // Retrieve data from GPU to CPU asynchronously
    bool RetrieveDataFromGPU(std::vector<float>& data, const std::string& task);

    // Generate random vertices for meshes (used in rendering, physics, AI)
    bool GenerateRandomVertices(std::vector<float>& vertices, size_t vertexCount);

    // Batch processing for large datasets
    void ProcessBatch(const std::vector<std::vector<float>>& batchData, const std::string& task);

private:
    size_t bufferSize;       // Size of the data buffer
    float* pinnedMemory;     // Pinned memory on the CPU
    float* deviceMemory;     // Device memory on the GPU
    cudaStream_t stream;     // CUDA stream for asynchronous operations

    // Task-specific kernel launches
    bool LaunchPhysicsKernel(const std::vector<float>& data);
    bool LaunchGraphicsKernel(const std::vector<float>& data);
    bool LaunchAIKernel(const std::vector<float>& data);
    bool LaunchVertexKernel(float* deviceVertices, size_t vertexCount);

    // Utility functions
    void LogTask(const std::string& task);
    void CheckCudaError(const std::string& message); // Error handling
};

#endif // CPUGPU_DATA_PIPELINE_H
