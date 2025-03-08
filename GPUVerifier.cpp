#include "GPUVerifier.h"
#include <cuda_runtime.h>
#include <d3d12.h>      // Direct3D 12 API
#include <dxgi1_6.h>    // DXGI API for GPU enumeration
#include <wrl.h>        // Microsoft WRL for COM objects
#include <iostream>

using namespace Microsoft::WRL; // For smart COM pointers

GPUVerifier::GPUVerifier() {
    RetrieveGPUInfo();
}

void GPUVerifier::RetrieveGPUInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Initialize DXGI for DirectX GPU enumeration
    ComPtr<IDXGIFactory6> dxgiFactory;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory6), reinterpret_cast<void**>(dxgiFactory.GetAddressOf()));

    if (FAILED(hr)) {
        std::cerr << "Failed to create DXGIFactory for DirectX detection!" << std::endl;
        return;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        GPUInfo gpuInfo;
        gpuInfo.name = deviceProp.name;
        gpuInfo.deviceID = device;
        gpuInfo.totalMemory = deviceProp.totalGlobalMem;
        gpuInfo.computeCapabilityMajor = deviceProp.major;
        gpuInfo.computeCapabilityMinor = deviceProp.minor;
        gpuInfo.supportsCUDA = true; // All detected GPUs support CUDA
        gpuInfo.supportsDirectX = false; // Default to false

        // Check DirectX Support for this GPU
        ComPtr<IDXGIAdapter1> dxgiAdapter;
        for (UINT adapterIndex = 0; dxgiFactory->EnumAdapters1(adapterIndex, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND; ++adapterIndex) {
            DXGI_ADAPTER_DESC1 adapterDesc;
            dxgiAdapter->GetDesc1(&adapterDesc);

            std::wstring wName(adapterDesc.Description);
            std::string adapterName(wName.begin(), wName.end()); // Convert to std::string

            if (adapterName.find(gpuInfo.name) != std::string::npos) {
                // Try creating a Direct3D12 device to confirm DX support
                ComPtr<ID3D12Device> d3d12Device;
                if (SUCCEEDED(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
                    gpuInfo.supportsDirectX = true;
                }
                break;
            }
        }

        gpuList.push_back(gpuInfo);
    }
}

std::vector<GPUInfo> GPUVerifier::GetAvailableGPUs() const {
    return gpuList;
}

void GPUVerifier::LogGPUDetails() const {
    for (const auto& gpu : gpuList) {
        std::cout << "GPU Name: " << gpu.name << std::endl;
        std::cout << "Device ID: " << gpu.deviceID << std::endl;
        std::cout << "Total Memory: " << gpu.totalMemory / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Compute Capability: " << gpu.computeCapabilityMajor << "." << gpu.computeCapabilityMinor << std::endl;
        std::cout << "Supports CUDA: " << (gpu.supportsCUDA ? "Yes" : "No") << std::endl;
        std::cout << "Supports DirectX: " << (gpu.supportsDirectX ? "Yes" : "No") << std::endl;
        std::cout << "--------------------------" << std::endl;
    }
}
