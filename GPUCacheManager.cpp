#include "GPUCacheManager.h"

GPUCacheManager::GPUCacheManager() {}
GPUCacheManager::~GPUCacheManager() {}

bool GPUCacheManager::Initialize() {
    std::cout << "Initializing GPU Cache Manager..." << std::endl;
    ScanCacheDirectory();  // Ensure cache is scanned on startup
    return hyperCache.InitializeCache();
}

bool GPUCacheManager::ExtractAndCacheData(ID3D12GraphicsCommandList* cmdList, const std::string& applicationName) {
    std::vector<float> motionVectors = ExtractMotionVectors(cmdList);
    return SaveFrameDataToCache(motionVectors, applicationName + "_dx12.cache");
}

bool GPUCacheManager::ExtractAndCacheData(ID3D11DeviceContext* deviceContext, const std::string& applicationName) {
    std::vector<float> motionVectors = ExtractMotionVectors(deviceContext);
    return SaveFrameDataToCache(motionVectors, applicationName + "_dx11.cache");
}

bool GPUCacheManager::SaveFrameDataToCache(const std::vector<float>& data, const std::string& filename) {
    std::string filePath = cacheDirectory + "\\" + filename;
    std::ofstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to save cache file: " << filePath << std::endl;
        return false;
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    file.close();
    return true;
}

bool GPUCacheManager::LoadCacheForApplication(const std::string& applicationName) {
    std::vector<std::string> cacheFiles = GetCacheFilesForApplication(applicationName);
    if (cacheFiles.empty()) {
        std::cout << "No cache files found for " << applicationName << std::endl;
        return false;
    }

    for (const auto& file : cacheFiles) {
        std::cout << "Loading cache file: " << file << std::endl;
        // Implement file loading logic here
    }
    return true;
}

void GPUCacheManager::ScanCacheDirectory() {
    std::cout << "Scanning cache directory: " << cacheDirectory << std::endl;

    if (!std::filesystem::exists(cacheDirectory)) {
        std::cout << "Cache directory does not exist. Creating it..." << std::endl;
        std::filesystem::create_directories(cacheDirectory);
    }

    for (const auto& entry : std::filesystem::directory_iterator(cacheDirectory)) {
        std::string filePath = entry.path().string();
        std::cout << "Found cache file: " << filePath << std::endl;

        // Extract application name from cache file and check if it has associated data
        size_t start = filePath.find_last_of("/\\") + 1;
        size_t end = filePath.find("_dx"); // Detect the format "_dx12.cache" or "_dx11.cache"
        if (end != std::string::npos) {
            std::string appName = filePath.substr(start, end - start);
            std::vector<std::string> appCacheFiles = GetCacheFilesForApplication(appName);
            std::cout << "Cache linked to application: " << appName << " (" << appCacheFiles.size() << " files found)" << std::endl;
        }
    }
}

std::vector<float> GPUCacheManager::ExtractMotionVectors(ID3D12GraphicsCommandList* cmdList) {
    // Placeholder: Implement DX12 motion vector extraction
    return std::vector<float>(100, 0.5f);
}

std::vector<float> GPUCacheManager::ExtractMotionVectors(ID3D11DeviceContext* deviceContext) {
    // Placeholder: Implement DX11 motion vector extraction
    return std::vector<float>(100, 0.5f);
}

bool GPUCacheManager::CacheExistsForApplication(const std::string& applicationName) {
    return !GetCacheFilesForApplication(applicationName).empty();
}

std::vector<std::string> GPUCacheManager::GetCacheFilesForApplication(const std::string& applicationName) {
    std::vector<std::string> cacheFiles;
    for (const auto& entry : std::filesystem::directory_iterator(cacheDirectory)) {
        if (entry.path().string().find(applicationName) != std::string::npos) {
            cacheFiles.push_back(entry.path().string());
        }
    }
    return cacheFiles;
}
