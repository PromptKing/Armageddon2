#ifndef VRAM_CACHE_MANAGER_H
#define VRAM_CACHE_MANAGER_H

#include <d3d12.h>
#include <d3d11.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "MemoryBridge.h"

class VRAMCacheManager {
public:
    VRAMCacheManager();
    ~VRAMCacheManager();

    // Initialize VRAM caching system (Only if DX11 or DX12 is detected)
    bool Initialize(size_t maxCacheSizeMB = 1024);

    // Store data in VRAM cache
    bool StoreDataInVRAM(const std::string& key, const std::vector<float>& data);

    // Retrieve cached data from VRAM
    bool RetrieveDataFromVRAM(const std::string& key, std::vector<float>& data);

    // Release VRAM on shutdown
    void ReleaseVRAM();

private:
    size_t maxCacheSize;
    size_t currentCacheSize;
    std::unordered_map<std::string, void*> vramCache;
    std::unordered_map<std::string, size_t> vramCacheSize;

    // GPU buffer allocation helpers
    bool AllocateVRAMBuffer(void** buffer, size_t size);
    void FreeVRAMBuffer(void* buffer);

    // Detects if DirectX 11 or 12 is available
    bool DetectDirectXVersion();
};

#endif // VRAM_CACHE_MANAGER_H
