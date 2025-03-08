
#include <cuda_runtime.h>
#include <d3d12.h>
#include <iostream>
#include "cuda_d3d12_interop.h"

// Maps a Direct3D 12 resource to CUDA for processing
bool MapD3D12ResourceToCUDA(ID3D12Resource* d3dResource, cudaGraphicsResource_t& cudaResource) {
    cudaError_t cudaStatus;

    // Register the DirectX 12 resource with CUDA
    cudaStatus = cudaGraphicsD3D12RegisterResource(&cudaResource, d3dResource, cudaGraphicsRegisterFlagsNone);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error: Failed to register D3D12 resource with CUDA. "
            << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }

    // Map the resource so it can be accessed by CUDA
    cudaStatus = cudaGraphicsMapResources(1, &cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error: Failed to map D3D12 resource to CUDA. "
            << cudaGetErrorString(cudaStatus) << std::endl;

        // Unregister the resource in case of failure
        cudaGraphicsUnregisterResource(cudaResource);
        return false;
    }

    std::cout << "Successfully mapped D3D12 resource to CUDA." << std::endl;
    return true;
}
