#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include "Metadata.h"

class MemoryBridge {
public:
    // Constructor and Destructor
    MemoryBridge(size_t bufferSize = 4092 * 4092);
    ~MemoryBridge();

    // Initialization and resource management
    bool Initialize();
    void ReleaseResources();

    // Data transfer methods
    bool CopyDataToGPU(const std::vector<float>& data);
    bool RetrieveDataFromGPU(std::vector<float>& data);

    // Metadata management
    void CacheMetadata(int threadID, const Metadata& metadata);
    bool RetrieveMetadata(int threadID, Metadata& metadata);

    // Utility functions
    static double GetCurrentTimestamp();

private:
    size_t bufferSize; // Size of the memory buffer
    std::unordered_map<int, Metadata> metadataCache; // Cache for metadata
    std::mutex cacheMutex; // Mutex to ensure thread safety
    void* gpuBuffer; // Pointer to the GPU memory buffer
};

