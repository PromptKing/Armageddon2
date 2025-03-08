#include "DataPipelineManager.h"

DataPipelineManager::DataPipelineManager(size_t bufferSize)
    : memoryBridge(bufferSize) {
}

DataPipelineManager::~DataPipelineManager() {
    ShutdownPipeline();
}

bool DataPipelineManager::InitializePipeline() {
    EnsureCacheDirectory();

    if (!memoryBridge.Initialize()) {
        std::cerr << "Failed to initialize MemoryBridge.\n";
        return false;
    }

    std::cout << "Pipeline initialized successfully.\n";
    return true;
}

bool DataPipelineManager::ExchangeData(const std::vector<float>& cpuData, std::vector<float>& gpuResult) {
    if (!memoryBridge.CopyDataToGPU(cpuData)) {
        std::cerr << "Failed to copy data to GPU.\n";
        return false;
    }

    if (!memoryBridge.RetrieveDataFromGPU(gpuResult)) {
        std::cerr << "Failed to retrieve data from GPU.\n";
        return false;
    }

    // Cache the exchanged data
    CacheData("latest_cpu_data.cache", cpuData);
    CacheData("latest_gpu_result.cache", gpuResult);

    return true;
}

void DataPipelineManager::CacheData(const std::string& filename, const std::vector<float>& data) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    std::string filepath = std::string(CACHE_DIR) + "\\" + filename;

    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open cache file for writing: " << filepath << "\n";
        return;
    }

    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    file.close();

    std::cout << "Data cached to: " << filepath << "\n";
}

bool DataPipelineManager::RetrieveCachedData(const std::string& filename, std::vector<float>& data) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    std::string filepath = std::string(CACHE_DIR) + "\\" + filename;

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open cache file for reading: " << filepath << "\n";
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    data.resize(fileSize / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();

    std::cout << "Data retrieved from cache: " << filepath << "\n";
    return true;
}

void DataPipelineManager::ShutdownPipeline() {
    memoryBridge.ReleaseResources();
    std::cout << "Pipeline shut down and resources released.\n";
}

void DataPipelineManager::EnsureCacheDirectory() {
    std::lock_guard<std::mutex> lock(cacheMutex);

    if (!std::filesystem::exists(CACHE_DIR)) {
        if (!std::filesystem::create_directory(CACHE_DIR)) {
            std::cerr << "Failed to create cache directory: " << CACHE_DIR << "\n";
        }
        else {
            std::cout << "Cache directory created at: " << CACHE_DIR << "\n";
        }
    }
}
