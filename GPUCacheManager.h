#ifndef GPU_CACHE_MANAGER_H
#define GPU_CACHE_MANAGER_H

#include <d3d12.h>
#include <d3d11.h>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "HyperCache.h"
#include "Metadata.h"
#include "MemoryBridge.h"

class GPUCacheManager {
public:
    GPUCacheManager();
    ~GPUCacheManager();

    // Initialize cache system
    bool Initialize();

    // Extract and cache GPU data (motion vectors, images, textures)
    bool ExtractAndCacheData(ID3D12GraphicsCommandList* cmdList, const std::string& applicationName);
    bool ExtractAndCacheData(ID3D11DeviceContext* deviceContext, const std::string& applicationName);

    // Load cached data at startup
    bool LoadCacheForApplication(const std::string& applicationName);

    // Scan cache directory for available cache files
    void ScanCacheDirectory();

private:
    HyperCache hyperCache;
    MemoryBridge memoryBridge;
    std::string cacheDirectory = "F:\\AI\\Arma Cache";

    // Helper function to save extracted frame data
    bool SaveFrameDataToCache(const std::vector<float>& data, const std::string& filename);

    // Helper function to extract motion vectors
    std::vector<float> ExtractMotionVectors(ID3D12GraphicsCommandList* cmdList);
    std::vector<float> ExtractMotionVectors(ID3D11DeviceContext* deviceContext);

    // Utility function to check if cache exists for an application
    bool CacheExistsForApplication(const std::string& applicationName);

    // Utility function to load cache files
    std::vector<std::string> GetCacheFilesForApplication(const std::string& applicationName);
};

#endif // GPU_CACHE_MANAGER_H
