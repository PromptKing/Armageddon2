#include "VRAMCacheManager.h"
#include <dxgi.h>
#include <iostream>
#include <windows.h>
#include <cuda_runtime.h>
#include <d3d12.h>
#include "d3d11.h"


VRAMCacheManager::VRAMCacheManager() : maxCacheSize(1024), currentCacheSize(0) {}
VRAMCacheManager::~VRAMCacheManager() {
    ReleaseVRAM();
}

bool VRAMCacheManager::Initialize(size_t maxCacheSizeMB) {
    if (!DetectDirectXVersion()) {
        std::cerr << "DirectX 11 or 12 not detected! VRAM Cache will not be used." << std::endl;
        return false;
    }

    maxCacheSize = maxCacheSizeMB;
    currentCacheSize = 0;
    std::cout << "VRAM Cache initialized with max size: " << maxCacheSize << " MB" << std::endl;
    return true;
}

bool VRAMCacheManager::DetectDirectXVersion() {
    IDXGIFactory* factory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory))) {
        std::cerr << "Failed to create DXGI Factory. DirectX may not be available." << std::endl;
        return false;
    }

    IDXGIAdapter* adapter = nullptr;
    if (FAILED(factory->EnumAdapters(0, &adapter))) {
        std::cerr << "No DirectX-compatible GPU detected!" << std::endl;
        factory->Release();
        return false;
    }

    DXGI_ADAPTER_DESC desc;
    adapter->GetDesc(&desc);
    std::wcout << L"Detected GPU: " << desc.Description << std::endl;

    bool dx12Supported = false, dx11Supported = false;

    // Check if DirectX 12 is supported
    ID3D12Device* d3d12Device = nullptr;
    if (SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), (void**)&d3d12Device))) {
        std::cout << "DirectX 12 is supported!" << std::endl;
        dx12Supported = true;
        d3d12Device->Release();
    }

    // Check if DirectX 11 is supported
    ID3D11Device* d3d11Device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    if (SUCCEEDED(D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
        D3D11_SDK_VERSION, &d3d11Device, nullptr, &context))) {
        std::cout << "DirectX 11 is supported!" << std::endl;
        dx11Supported = true;
        d3d11Device->Release();
        context->Release();
    }

    adapter->Release();
    factory->Release();

    return dx12Supported || dx11Supported;
}

bool VRAMCacheManager::StoreDataInVRAM(const std::string& key, const std::vector<float>& data) {
    size_t dataSize = data.size() * sizeof(float);
    if (currentCacheSize + dataSize > maxCacheSize * 1024 * 1024) {
        std::cerr << "VRAM Cache limit exceeded! Cannot store " << key << std::endl;
        return false;
    }

    void* deviceBuffer;
    if (!AllocateVRAMBuffer(&deviceBuffer, dataSize)) return false;

    cudaMemcpy(deviceBuffer, data.data(), dataSize, cudaMemcpyHostToDevice);
    vramCache[key] = deviceBuffer;
    vramCacheSize[key] = dataSize;
    currentCacheSize += dataSize;

    std::cout << "Stored " << key << " in VRAM (Size: " << dataSize / (1024.0 * 1024.0) << " MB)" << std::endl;
    return true;
}

bool VRAMCacheManager::RetrieveDataFromVRAM(const std::string& key, std::vector<float>& data) {
    if (vramCache.find(key) == vramCache.end()) {
        std::cerr << "No VRAM cache found for key: " << key << std::endl;
        return false;
    }

    size_t dataSize = vramCacheSize[key];
    data.resize(dataSize / sizeof(float));
    cudaMemcpy(data.data(), vramCache[key], dataSize, cudaMemcpyDeviceToHost);

    std::cout << "Retrieved " << key << " from VRAM" << std::endl;
    return true;
}

void VRAMCacheManager::ReleaseVRAM() {
    for (auto& entry : vramCache) {
        FreeVRAMBuffer(entry.second);
    }
    vramCache.clear();
    vramCacheSize.clear();
    currentCacheSize = 0;
    std::cout << "Released all VRAM Cache." << std::endl;
}

bool VRAMCacheManager::AllocateVRAMBuffer(void** buffer, size_t size) {
    if (cudaMalloc(buffer, size) != cudaSuccess) {
        std::cerr << "Failed to allocate VRAM buffer of size: " << size << " bytes" << std::endl;
        return false;
    }
    return true;
}

void VRAMCacheManager::FreeVRAMBuffer(void* buffer) {
    cudaFree(buffer);
}
