#include "cuda_d3d12_interop.h"
#include <cuda_runtime.h>
#include <iostream>
#include <d3d12.h>
#include <windows.h>
#include <device_launch_parameters.h>
#include <vector>
#include <set> // Include for std::set
#include <chrono>
#include <thread>
#include <TlHelp32.h>
#include <mutex> // For GPU and CPU communication
#include <iomanip> // For formatted output
#include <string>
#include <fstream>
#include <cmath> // For std::abs
#include <queue>
#include <sstream> // Include for std::ostringstream
#include "utility.h"
#include <cuda_runtime_api.h> // Updated include
#include "CodeInjector.h"
#include "GPUVerifier.h"
#include "HostVerifier.h"
#include "CUBLASManager.h"
#include <cublas_v2.h>
#include <driver_types.h>
#include "ThreadMapper.h"
#include "MatrixOps.h"
#include "MemoryBridge.h"
#include "Metadata.h"
#include "PinnedMemory.h"
#include "CUDA_SPY.h" 
#include "CudaUtils.h"
#include "SupervisorData.h"
#include "ArmageddonAlgorithm.h"



// Extern "C" declarations for CUDA functions
extern "C" cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
extern "C" cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream);
extern "C" cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource);
extern "C" cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream);
extern "C" cudaError_t cudaGraphicsD3D12RegisterResource(cudaGraphicsResource_t* resource, ID3D12Resource* d3d12Resource, unsigned int flags);

// Function to initialize and register a D3D12 resource with CUDA
bool InitializeD3D12ResourceWithCUDA(ID3D12Resource* d3d12Resource) {
    cudaGraphicsResource_t cudaResource;
    cudaError_t cudaStatus;

    // Register the Direct3D 12 resource with CUDA
    cudaStatus = cudaGraphicsD3D12RegisterResource(&cudaResource, d3d12Resource, cudaGraphicsRegisterFlagsNone);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to register D3D12 resource with CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }

    // Map the resource for CUDA
    cudaStatus = cudaGraphicsMapResources(1, &cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to map D3D12 resource with CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaGraphicsUnregisterResource(cudaResource);
        return false;
    }

    void* cudaPtr = nullptr;
    size_t size = 0;

    // Get the device pointer to the resource
    cudaStatus = cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to get mapped pointer: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaGraphicsUnmapResources(1, &cudaResource);
        cudaGraphicsUnregisterResource(cudaResource);
        return false;
    }

    std::cout << "D3D12 resource successfully registered with CUDA." << std::endl;
    std::cout << "CUDA device pointer: " << cudaPtr << ", Size: " << size << " bytes" << std::endl;

    // Unmap the resource
    cudaStatus = cudaGraphicsUnmapResources(1, &cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to unmap D3D12 resource: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaGraphicsUnregisterResource(cudaResource);
        return false;
    }

    // Unregister the resource after usage (if no longer needed)
    cudaStatus = cudaGraphicsUnregisterResource(cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to unregister D3D12 resource: " << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }

    return true;
}

// CUDA API mock implementation for interop with D3D12 (for completeness in absence of real headers)

extern "C" cudaError_t cudaGraphicsD3D12RegisterResource(cudaGraphicsResource_t* resource, ID3D12Resource* d3d12Resource, unsigned int flags) {
    if (!d3d12Resource) {
        std::cerr << "Error: Invalid Direct3D12 resource.\n";
        return cudaErrorInvalidValue;
    }

    // Retrieve resource description for debugging and validation
    D3D12_RESOURCE_DESC desc = d3d12Resource->GetDesc();
    std::cout << "Registering D3D12 resource with CUDA:\n"
        << "  Width: " << desc.Width << "\n"
        << "  Height: " << desc.Height << "\n"
        << "  Format: " << desc.Format << "\n";

    // Mocking the behavior of associating a CUDA graphics resource
    *resource = reinterpret_cast<cudaGraphicsResource_t>(d3d12Resource);

    // Check if the resource is suitable for CUDA processing
    if (!(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
        std::cerr << "Error: Resource does not allow unordered access.\n";
        return cudaErrorInvalidValue;
    }

    std::cout << "Resource is valid for CUDA registration.\n";

    // Motion vector processing for generating new frames
    cudaError_t cudaStatus;
    cudaGraphicsResource_t cudaResource;

    // Register the resource with CUDA
    cudaStatus = cudaGraphicsD3D12RegisterResource(&cudaResource, d3d12Resource, flags);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to register D3D12 resource with CUDA: " << cudaGetErrorString(cudaStatus) << "\n";
        return cudaStatus;
    }

    // Map the resource to CUDA memory
    cudaStatus = cudaGraphicsMapResources(1, &cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to map D3D12 resource to CUDA: " << cudaGetErrorString(cudaStatus) << "\n";
        cudaGraphicsUnregisterResource(cudaResource);
        return cudaStatus;
    }

    // Obtain device pointer for processing
    void* devPtr;
    size_t size;
    cudaStatus = cudaGraphicsResourceGetMappedPointer(&devPtr, &size, cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to get mapped pointer from resource: " << cudaGetErrorString(cudaStatus) << "\n";
        cudaGraphicsUnmapResources(1, &cudaResource);
        cudaGraphicsUnregisterResource(cudaResource);
        return cudaStatus;
    }

    std::cout << "CUDA resource mapped successfully. Processing motion vectors...\n";

    

    // Unmap the resource
    cudaStatus = cudaGraphicsUnmapResources(1, &cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to unmap CUDA resource: " << cudaGetErrorString(cudaStatus) << "\n";
        cudaGraphicsUnregisterResource(cudaResource);
        return cudaStatus;
    }

    std::cout << "Real-time frame generation based on motion vectors completed.\n";

    return cudaSuccess;
}

__global__ void ProcessMotionVectors(void* devPtr, unsigned int width, unsigned int height) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) return;

    // Compute the linear index
    unsigned int index = idy * width + idx;

    // Example motion vector computation (replace with actual logic)
    float motionVector = sinf(idx * 0.01f) + cosf(idy * 0.01f);

    // Update the pixel data (e.g., for the next frame)
    float* pixelData = reinterpret_cast<float*>(devPtr);
    pixelData[index] = motionVector;
}


extern "C" cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
    // Mocked implementation. Replace with actual CUDA library call.
    std::cout << "Mock: Unregistering CUDA resource.\n";
    return cudaSuccess;
}

extern "C" cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
    // Mocked implementation. Replace with actual CUDA library call.
    std::cout << "Mock: Mapping CUDA resources.\n";
    return cudaSuccess;
}

extern "C" cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
    // Mocked implementation. Replace with actual CUDA library call.
    std::cout << "Mock: Unmapping CUDA resources.\n";
    return cudaSuccess;
}

extern "C" cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) {
    // Mocked implementation. Replace with actual CUDA library call.
    std::cout << "Mock: Getting mapped pointer for CUDA resource.\n";
    *devPtr = resource; // This assumes the resource itself is a pointer in this mock
    *size = static_cast<size_t>(4092) * 4092; // Example size, replace with actual resource size
    return cudaSuccess;
}
