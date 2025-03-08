#ifndef DATA_PIPELINE_MANAGER_H
#define DATA_PIPELINE_MANAGER_H

#include "MemoryBridge.h"
#include <vector>
#include <string>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "DataPipelineManager.h"

// Ensure the cache directory exists
constexpr const char* CACHE_DIR = "F:\\MemoryBridgeCache";

class DataPipelineManager {
public:
    DataPipelineManager(size_t bufferSize = 1024 * 1024);
    ~DataPipelineManager();

    // Initialize pipeline and resources
    bool InitializePipeline();

    // Exchange data between CPU and GPU
    bool ExchangeData(const std::vector<float>& cpuData, std::vector<float>& gpuResult);

    // Cache management
    void CacheData(const std::string& filename, const std::vector<float>& data);
    bool RetrieveCachedData(const std::string& filename, std::vector<float>& data);

    // Release resources
    void ShutdownPipeline();

private:
    MemoryBridge memoryBridge;
    std::mutex cacheMutex;

    void EnsureCacheDirectory();
};

#endif // DATA_PIPELINE_MANAGER_H
