#ifndef CUDA_D3D12_INTEROP_H
#define CUDA_D3D12_INTEROP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <d3d12.h>

typedef struct cudaGraphicsResource* D3D12CUDAResource;

// Define CUDA-Direct3D 12 interop flags
#define CUDA_GRAPHICS_REGISTER_FLAGS_NONE 0
#define CUDA_GRAPHICS_REGISTER_FLAGS_READ_ONLY 1
#define CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD 2

// CUDA Graphics Resource Handle
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

// CUDA Error Handling
#ifdef __cplusplus
extern "C" {
#endif

    // Register a Direct3D 12 resource for access by CUDA
    cudaError_t cudaGraphicsD3D12RegisterResource(
        cudaGraphicsResource_t* resource,         // Pointer to the resource handle
        ID3D12Resource* d3d12Resource,           // Pointer to the D3D12 resource
        unsigned int flags                        // Register flags
    );

    // Unregister a Direct3D 12 resource
    cudaError_t cudaGraphicsUnregisterResource(
        cudaGraphicsResource_t resource           // Resource handle to unregister
    );

    // Map resources for CUDA access
    cudaError_t cudaGraphicsMapResources(
        int count,                                // Number of resources to map
        cudaGraphicsResource_t* resources,       // Array of resource handles
        cudaStream_t stream                       // CUDA stream for the operation
    );

    // Unmap resources after CUDA access
    cudaError_t cudaGraphicsUnmapResources(
        int count,                                // Number of resources to unmap
        cudaGraphicsResource_t* resources,       // Array of resource handles
        cudaStream_t stream                       // CUDA stream for the operation
    );

    // Get a device pointer for a subresource in a mapped graphics resource
    cudaError_t cudaGraphicsResourceGetMappedPointer(
        void** devPtr,                            // Pointer to the device memory
        size_t* size,                             // Size of the mapped memory
        cudaGraphicsResource_t resource           // Resource handle
    );

#ifdef __cplusplus
}
#endif

#endif // CUDA_D3D12_INTEROP_H
