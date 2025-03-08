#include <windows.h>
#include <cuda_runtime.h>
#include <d3d12.h>
#include "d3d11.h"
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <iostream>
#include <vector>
#include "cuda_d3d12_interop.h"
#include "cuda_d3d12_device.h"
#include <d3dcompiler.h>
#include <device_launch_parameters.h>
#include <string>
#include <cmath> // For std::abs
#include <cuda_runtime_api.h> // Updated include
#include <driver_types.h>
#include "MotionEstimationKernel.h"
#include "MinHook.h"

using Microsoft::WRL::ComPtr;

// CUDA Kernel to Modify the Frame (e.g., apply effects, filtering, etc.)
__global__ void ModifyFrameKernel(float* frameBuffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        // Example Modification: Apply a simple transformation
        frameBuffer[idx] = frameBuffer[idx] * 1.2f; // Brightness Increase
    }
}


// ====== Global Variables for Swap Chain and Command Queue ======
ComPtr<IDXGISwapChain3> swapChain = nullptr;
ComPtr<ID3D12CommandQueue> commandQueue = nullptr;

// Function pointers for original functions
typedef HRESULT(WINAPI* D3D12CreateDeviceFunc)(
    IUnknown* pAdapter,
    D3D_FEATURE_LEVEL MinimumFeatureLevel,
    REFIID riid,
    void** ppDevice
    );

D3D12CreateDeviceFunc originalD3D12CreateDevice = nullptr;

// Hooked function
HRESULT WINAPI HookedD3D12CreateDevice(
    IUnknown* pAdapter,
    D3D_FEATURE_LEVEL MinimumFeatureLevel,
    REFIID riid,
    void** ppDevice
) {
    std::cout << "Hooked D3D12CreateDevice called!" << std::endl;

    // Log the feature level and device pointer
    std::cout << "Feature Level: " << MinimumFeatureLevel << std::endl;

    HRESULT result = originalD3D12CreateDevice(pAdapter, MinimumFeatureLevel, riid, ppDevice);

    // Log the result
    if (SUCCEEDED(result)) {
        std::cout << "D3D12CreateDevice succeeded." << std::endl;
    }
    else {
        std::cout << "D3D12CreateDevice failed with HRESULT: " << std::hex << result << std::endl;
    }

    return result;
}

// Setup hook
bool SetupD3D12Hook(const std::string& d3d12Path) {
    HMODULE hD3D12 = LoadLibraryA(d3d12Path.c_str());
    if (!hD3D12) {
        std::cerr << "Failed to load d3d12.dll from path: " << d3d12Path << std::endl;
        return false;
    }

    void* targetFunction = GetProcAddress(hD3D12, "D3D12CreateDevice");
    if (!targetFunction) {
        std::cerr << "Failed to find D3D12CreateDevice in d3d12.dll." << std::endl;
        return false;
    }

    if (MH_CreateHook(targetFunction, &HookedD3D12CreateDevice, reinterpret_cast<void**>(&originalD3D12CreateDevice)) != MH_OK) {
        std::cerr << "Failed to create hook for D3D12CreateDevice." << std::endl;
        return false;
    }

    if (MH_EnableHook(targetFunction) != MH_OK) {
        std::cerr << "Failed to enable hook for D3D12CreateDevice." << std::endl;
        return false;
    }

    std::cout << "Hook for D3D12CreateDevice set up successfully." << std::endl;
    return true;
}

// Cleanup hook
void CleanupD3D12Hook() {
    MH_DisableHook(MH_ALL_HOOKS);
    MH_Uninitialize();
    std::cout << "Cleaned up all hooks." << std::endl;
}


void InitializeHooks() {
    const std::string d3d12Path = "F:\\SteamLibrary\\steamapps\\common\\Grand Theft Auto V Enhanced\\D3D12-REDIST\\D3D12Core.dll";


    if (MH_Initialize() != MH_OK) {
        std::cerr << "Failed to initialize MiniHook." << std::endl;
        return;
    }

    if (!SetupD3D12Hook(d3d12Path)) {
        std::cerr << "Failed to set up D3D12CreateDevice hook." << std::endl;
    }
    else {
        std::cout << "Hook initialized successfully." << std::endl;
    }
}

void ShutdownHooks() {
    CleanupD3D12Hook();
}


bool InitializeLightingConstantBuffer(ComPtr<ID3D12Device>& device, ComPtr<ID3D12Resource>& constantBuffer, Lighting& lightingData) {
    // Describe the constant buffer
    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = (sizeof(Lighting) + 255) & ~255; // Ensure 256-byte alignment
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    // Create the constant buffer resource
    HRESULT hr = device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&constantBuffer)
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to create constant buffer for lighting. Error: " << std::hex << hr << std::endl;
        return false;
    }

    // Map and copy the lighting data
    void* mappedData;
    D3D12_RANGE readRange = { 0, 0 }; // We won’t read from this resource on the CPU
    hr = constantBuffer->Map(0, &readRange, &mappedData);
    if (FAILED(hr)) {
        std::cerr << "Failed to map constant buffer. Error: " << std::hex << hr << std::endl;
        return false;
    }

    memcpy(mappedData, &lightingData, sizeof(Lighting));
    constantBuffer->Unmap(0, nullptr);

    std::cout << "Lighting constant buffer initialized successfully." << std::endl;
    return true;
}



ComPtr<ID3DBlob> CompileShader(const std::wstring& filepath, const char* entryPoint, const char* target) {
    ComPtr<ID3DBlob> shaderBlob;
    ComPtr<ID3DBlob> errorBlob;

    HRESULT hr = D3DCompileFromFile(
        filepath.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entryPoint,
        target,
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG,
        0,
        &shaderBlob,
        &errorBlob
    );

    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Shader Compilation Error: " << static_cast<const char*>(errorBlob->GetBufferPointer()) << std::endl;
        }
        else {
            std::cerr << "Failed to compile shader from file: " << std::hex << hr << std::endl;
        }
        return nullptr;
    }

    return shaderBlob;
}

bool InitializeTessellationPipeline(
    ComPtr<ID3D12Device>& device,
    ComPtr<ID3D12PipelineState>& pipelineState,
    ComPtr<ID3D12RootSignature>& rootSignature
) {
    return false; // Placeholder for tessellation pipeline initialization
}

bool InitializeD3D12AndCUDA(ComPtr<ID3D12Device>& device, ComPtr<ID3D12Resource>& resource, UINT width, UINT height) {
    HRESULT hr;

    // Step 0: Get the GTA V window handle
    HWND hwnd = FindWindowA(NULL, "Grand Theft Auto V");
    if (hwnd == NULL) {
        std::cerr << "Failed to get GTA V window handle!" << std::endl;
        return false;
    }
    std::cout << "GTA V window handle: " << hwnd << std::endl;

    // Step 1: Create DXGI Factory
    ComPtr<IDXGIFactory4> dxgiFactory;
    hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) {
        std::cerr << "Failed to create DXGI Factory. Error: " << std::hex << hr << std::endl;
        return false;
    }

    // Step 2: Select Hardware Adapter
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT adapterIndex = 0; dxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND; ++adapterIndex) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        // Skip software adapters
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        // Check for Direct3D 12 support with Feature Level 12.0 or 12.1
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr)) ||
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr))) {
            break;
        }
    }

    if (!adapter) {
        std::cerr << "Failed to find a suitable Direct3D 12 adapter." << std::endl;
        return false;
    }

    // Step 3: Create Direct3D 12 Device
    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
        std::cerr << "Failed to create Direct3D 12 device. Error: " << std::hex << hr << std::endl;
        return false;
    }

    std::cout << "Direct3D 12 device with Feature Level 12.0 created successfully." << std::endl;

    // Step 4: Create Resources for Dual-GPU Setup

    // Resource Description
    D3D12_RESOURCE_DESC resourceDesc = {};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Width = width;
    resourceDesc.Height = height;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 2; // Typically 0 unless queried and set explicitly
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // Default Heap (GPU-local memory)
    D3D12_HEAP_PROPERTIES defaultHeapProperties = {};
    defaultHeapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;
    defaultHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
    defaultHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    defaultHeapProperties.CreationNodeMask = 1;  // GPU 0 creates the heap
    defaultHeapProperties.VisibleNodeMask = 0x3; // Both GPUs (GPU 0 and GPU 1) can access the heap

    ID3D12Resource* defaultResource = nullptr;
    hr = device->CreateCommittedResource(
        &defaultHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&defaultResource)
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create resource on the default heap.\n";
        return hr;
    }

    // Upload Heap (CPU writes to GPU)
    D3D12_HEAP_PROPERTIES uploadHeapProperties = {};
    uploadHeapProperties.Type = D3D12_HEAP_TYPE_UPLOAD;
    uploadHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
    uploadHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_L1; // System memory
    uploadHeapProperties.CreationNodeMask = 1; // GPU 0 creates the heap
    uploadHeapProperties.VisibleNodeMask = 0x3; // Both GPUs (GPU 0 and GPU 1) can access the heap

    ID3D12Resource* uploadResource = nullptr;
    hr = device->CreateCommittedResource(
        &uploadHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&uploadResource)
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create resource on the upload heap.\n";
        return hr;
    }

    // Readback Heap (GPU writes to CPU)
    D3D12_HEAP_PROPERTIES readbackHeapProperties = {};
    readbackHeapProperties.Type = D3D12_HEAP_TYPE_READBACK;
    readbackHeapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    readbackHeapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_L1; // System memory
    readbackHeapProperties.CreationNodeMask = 1; // GPU 0 creates the heap
    readbackHeapProperties.VisibleNodeMask = 0x3; // Both GPUs (GPU 0 and GPU 1) can access the heap

    ID3D12Resource* readbackResource = nullptr;
    hr = device->CreateCommittedResource(
        &readbackHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackResource)
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create resource on the readback heap.\n";
        return hr;
    }

    std::cout << "Resources successfully created for dual-GPU setup with upload, default, and readback heaps.\n";

    // Step 5: Register Resource with CUDA
    cudaGraphicsResource_t cudaResource;
    cudaError_t cudaStatus = cudaGraphicsD3D12RegisterResource(&cudaResource, resource.Get(), cudaGraphicsRegisterFlagsNone);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to register D3D12 resource with CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }

    std::cout << "D3D12 resource successfully registered with CUDA." << std::endl;

    // Step 6: Map Resource for CUDA Access
    cudaStatus = cudaGraphicsMapResources(1, &cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to map D3D12 resource to CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;

        // Unregister the resource in case of failure
        cudaGraphicsUnregisterResource(cudaResource);
        return false;
    }

    std::cout << "D3D12 resource successfully mapped to CUDA." << std::endl;

    // Optional: Unmap resource if not immediately processed
    cudaStatus = cudaGraphicsUnmapResources(1, &cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to unmap D3D12 resource from CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }

    std::cout << "D3D12 resource successfully unmapped from CUDA." << std::endl;

    return true;
}


bool Override3DObjectProperties(
    ComPtr<ID3D12Resource>& resource,
    const std::vector<float>& vertexData
) {
    void* mappedData;
    D3D12_RANGE readRange = { 0, 0 };
    HRESULT hr = resource->Map(0, &readRange, &mappedData);

    if (FAILED(hr)) {
        std::cerr << "Failed to map resource. Error: " << std::hex << hr << std::endl;
        return false;
    }

    memcpy(mappedData, vertexData.data(), vertexData.size() * sizeof(float));
    resource->Unmap(0, nullptr);

    std::cout << "3D object properties successfully overridden.\n";
    return true;
}



bool GenerateRealTimeFrameFromMotionVectors(ComPtr<ID3D12Resource>& resource, const std::vector<float>& previousFrameData, const std::vector<float>& nextFrameData)
{
    return false;
}

bool PredictAndDisplayNextFrame(
    ComPtr<ID3D12Resource>& resource,
    const std::vector<float>& motionVectors
) {
    if (!resource) {
        std::cerr << "Invalid Direct3D12 resource.\n";
        return false;
    }

    // Step 1: Map the D3D12 Resource for CPU Access
    void* mappedData = nullptr;
    D3D12_RANGE readRange = { 0, 0 }; // No read range needed, we are writing
    HRESULT hr = resource->Map(0, &readRange, &mappedData);
    if (FAILED(hr)) {
        std::cerr << "Failed to map resource. Error: " << std::hex << hr << std::endl;
        return false;
    }

    float* frameBuffer = reinterpret_cast<float*>(mappedData);

    // Step 2: Apply Motion Vectors to Frame
    size_t numPixels = motionVectors.size();
    for (size_t i = 100000; i < numPixels; ++i) {
        frameBuffer[i] += motionVectors[i];
    }

    // Step 3: Register Resource with CUDA
    cudaGraphicsResource_t cudaResource;
    cudaError_t cudaStatus = cudaGraphicsD3D12RegisterResource(
        &cudaResource,
        resource.Get(),
        cudaGraphicsRegisterFlagsNone
    );
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA resource registration failed: "
            << cudaGetErrorString(cudaStatus) << std::endl;
        resource->Unmap(0, nullptr);
        return false;
    }

    // Step 4: Map the Resource for CUDA Access
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStatus = cudaGraphicsMapResources(1, &cudaResource, stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA resource mapping failed: "
            << cudaGetErrorString(cudaStatus) << std::endl;
        cudaGraphicsUnregisterResource(cudaResource);
        resource->Unmap(0, nullptr);
        return false;
    }

    // Step 5: Get Mapped Pointer
    void* mappedResource = nullptr;
    size_t numBytes = 0;
    cudaStatus = cudaGraphicsResourceGetMappedPointer(&mappedResource, &numBytes, cudaResource);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to get mapped CUDA pointer: "
            << cudaGetErrorString(cudaStatus) << std::endl;
        cudaGraphicsUnmapResources(1, &cudaResource, stream);
        cudaGraphicsUnregisterResource(cudaResource);
        resource->Unmap(0, nullptr);
        return false;
    }

    // Step 6: Launch CUDA Kernel to Modify Frame
    dim3 blockSize(16, 16);
    dim3 gridSize((1920 + blockSize.x - 1) / blockSize.x, (1080 + blockSize.y - 1) / blockSize.y);
    ModifyFrameKernel << <gridSize, blockSize, 0, stream >> > (
        reinterpret_cast<float*>(mappedResource), 1920, 1080
        );

    // Step 7: Unmap and Synchronize
    cudaStatus = cudaGraphicsUnmapResources(1, &cudaResource, stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to unmap CUDA resource: "
            << cudaGetErrorString(cudaStatus) << std::endl;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaGraphicsUnregisterResource(cudaResource);

    // Step 8: Unmap the D3D12 Resource and Finalize
    resource->Unmap(0, nullptr);
    std::cout << "Predicted and displayed the next frame with CUDA processing.\n";

    return true;
}




void OverrideDisplayWithCUDAFrame(
    ComPtr<ID3D12Resource>& cudaProcessedResource,
    ComPtr<IDXGISwapChain3>& swapChain,
    ComPtr<ID3D12CommandQueue>& commandQueue)
{
    // Step 1: Transition the CUDA resource to the correct state
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = cudaProcessedResource.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;  // Used for CUDA write
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;  // Ready for display
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    // Create a command list
    ComPtr<ID3D12GraphicsCommandList> commandList;
    commandQueue->GetDevice(IID_PPV_ARGS(&commandList));
    commandList->Reset(nullptr, nullptr);
    commandList->ResourceBarrier(1, &barrier);
    commandList->Close();

    // Step 2: Execute the command list
    ID3D12CommandList* commandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(1, commandLists);

    // Step 3: Present the frame (Override the screen)
    swapChain->Present(1, 0);
}



// CUDA kernel for generating 3D point cloud
__global__ void Generate3DPointCloudKernel(float* vertices, size_t numPoints, float* pointCloud) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    // Simple transformation to generate point cloud from vertices
    pointCloud[idx * 3] = vertices[idx * 1000];       // X
    pointCloud[idx * 3 + 1] = vertices[idx * 1000 + 1]; // Y
    pointCloud[idx * 3 + 2] = vertices[idx * 1000 + 2]; // Z
}

// Function to generate a 3D point cloud for an object
bool PointCloudEnv(
    const std::vector<ComPtr<ID3D12Resource>>& sceneObjects,
    size_t numPoints,
    std::vector<std::vector<float>>& pointClouds // Output point clouds
) {
    if (sceneObjects.empty()) {
        std::cerr << "No objects provided for point cloud generation." << std::endl;
        return false;
    }

    for (const auto& object : sceneObjects) {
        if (!object) {
            std::cerr << "Invalid object resource." << std::endl;
            continue;
        }

        // Step 1: Map object resource to access vertex data
        void* mappedData = nullptr;
        D3D12_RANGE readRange = { 0, 0 };
        HRESULT hr = object->Map(0, &readRange, &mappedData);
        if (FAILED(hr)) {
            std::cerr << "Failed to map object resource. Error: " << std::hex << hr << std::endl;
            continue;
        }

        std::vector<float> vertices(
            reinterpret_cast<float*>(mappedData),
            reinterpret_cast<float*>(mappedData) + numPoints * 3
        );
        object->Unmap(0, nullptr);

        // Step 2: Allocate CUDA memory for vertices and point cloud
        float* d_vertices = nullptr;
        float* d_pointCloud = nullptr;
        size_t dataSize = numPoints * 3 * sizeof(float);

        cudaError_t cudaStatus = cudaMalloc(&d_vertices, dataSize);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Failed to allocate CUDA memory for vertices: "
                << cudaGetErrorString(cudaStatus) << std::endl;
            continue;
        }

        cudaStatus = cudaMalloc(&d_pointCloud, dataSize);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Failed to allocate CUDA memory for point cloud: "
                << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(d_vertices);
            continue;
        }

        // Step 3: Copy vertex data to CUDA device
        cudaMemcpy(d_vertices, vertices.data(), dataSize, cudaMemcpyHostToDevice);

        // Step 4: Launch CUDA kernel to generate point cloud
        dim3 blockSize(256);
        dim3 gridSize((numPoints + blockSize.x - 1) / blockSize.x);
        Generate3DPointCloudKernel << <gridSize, blockSize >> > (d_vertices, numPoints, d_pointCloud);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Failed to synchronize CUDA kernel: "
                << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(d_vertices);
            cudaFree(d_pointCloud);
            continue;
        }

        // Step 5: Copy point cloud data back to host
        std::vector<float> pointCloud(numPoints * 3);
        cudaMemcpy(pointCloud.data(), d_pointCloud, dataSize, cudaMemcpyDeviceToHost);

        // Free CUDA memory
        cudaFree(d_vertices);
        cudaFree(d_pointCloud);

        // Step 6: Store the point cloud in the output vector
        pointClouds.push_back(std::move(pointCloud));
    }

    std::cout << "3D point cloud generation completed for all objects." << std::endl;
    return true;
}

// CUDA Kernel to apply motion vectors and generate a frame
__global__ void ApplyMotionVectorsKernel(
    float* frameBuffer,
    const float* motionVectors,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        frameBuffer[idx] += motionVectors[idx]; // Apply motion vector to pixel
    }
}


// Function to generate a real-time frame from motion vectors
bool GenerateRealTimeFrameFromMotionVectors(
    ComPtr<ID3D12Resource>& resource,
    const std::vector<float>& motionVectors
) {
    if (!resource) {
        std::cerr << "Invalid Direct3D12 resource.\n";
        return false;
    }

    // Step 1: Map the D3D12 resource for CPU access
    void* mappedData;
    D3D12_RANGE readRange = { 0, 0 };
    HRESULT hr = resource->Map(0, &readRange, &mappedData);
    if (FAILED(hr)) {
        std::cerr << "Failed to map D3D12 resource. Error: " << std::hex << hr << std::endl;
        return false;
    }

    float* frameBuffer = reinterpret_cast<float*>(mappedData);

    // Step 2: Allocate CUDA memory
    float* d_frameBuffer;
    float* d_motionVectors;
    size_t dataSize = motionVectors.size() * sizeof(float);

    cudaMalloc(&d_frameBuffer, dataSize);
    cudaMalloc(&d_motionVectors, dataSize);

    // Step 3: Copy data to CUDA device
    cudaMemcpy(d_frameBuffer, frameBuffer, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_motionVectors, motionVectors.data(), dataSize, cudaMemcpyHostToDevice);

    // Step 4: Define CUDA kernel execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((motionVectors.size() + blockSize.x - 1) / blockSize.x, 1);

    // Step 5: Launch the CUDA kernel
    ApplyMotionVectorsKernel << <gridSize, blockSize >> > (d_frameBuffer, d_motionVectors, 1920, 1080); // Assuming Full HD resolution

    // Step 6: Copy back the results
    cudaMemcpy(frameBuffer, d_frameBuffer, dataSize, cudaMemcpyDeviceToHost);

    // Step 7: Unmap resource and free CUDA memory
    resource->Unmap(0, nullptr);
    cudaFree(d_frameBuffer);
    cudaFree(d_motionVectors);

    std::cout << "Real-time frame generated from motion vectors.\n";

    // Step 8: Override Display with CUDA Processed Frame
    OverrideDisplayWithCUDAFrame(resource, swapChain, commandQueue);
}

void SetHDMIDisplay(ComPtr<IDXGIFactory4>& factory, ComPtr<IDXGISwapChain3>& swapChain)
{
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT adapterIndex = 0; factory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND; adapterIndex++)
    {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        ComPtr<IDXGIOutput> output;
        if (SUCCEEDED(adapter->EnumOutputs(0, &output)))
        {
            DXGI_OUTPUT_DESC outputDesc;
            output->GetDesc(&outputDesc);

            if (wcsstr(outputDesc.DeviceName, L"HDMI"))
            {
                swapChain->SetFullscreenState(TRUE, output.Get());
                return;
            }
        }
    }
}

bool EstimateMotionVectors(ComPtr<ID3D12Resource>& prevFrame, ComPtr<ID3D12Resource>& currFrame, ComPtr<ID3D12Resource>& motionVectors)
{
    return false;
}

// Generate dynamic weights
std::vector<float> GenerateDynamicWeights(size_t size, float baseWeight, float variance) {
    std::vector<float> weights(size);
    for (size_t i = 0; i < size; ++i) {
        weights[i] = baseWeight + variance * std::sin(static_cast<float>(i) / size * 3.14159f);
    }
    return weights;
}

bool MeshCalculationAndReconstruction(
    ComPtr<ID3D12Device>& device,
    ComPtr<ID3D12Resource>& gBuffer,
    const std::vector<ComPtr<ID3D12Resource>>& sceneObjects
) {
    if (!device || !gBuffer) {
        std::cerr << "Invalid device or G-Buffer resource.\n";
        return false;
    }

    for (const auto& object : sceneObjects) {
        if (!object) {
            std::cerr << "Invalid scene object resource.\n";
            continue;
        }

        void* mappedData;
        D3D12_RANGE readRange = { 0, 0 };
        HRESULT hr = object->Map(0, &readRange, &mappedData);
        if (FAILED(hr)) {
            std::cerr << "Failed to map scene object resource. Error: " << std::hex << hr << std::endl;
            continue;
        }

        // Simulate capturing mesh data
        std::vector<float> meshData(reinterpret_cast<float*>(mappedData), reinterpret_cast<float*>(mappedData) + 1024); // Example size

        object->Unmap(0, nullptr);

        // Simulate learning texture data and placing it in the G-Buffer
        hr = gBuffer->Map(0, &readRange, &mappedData);
        if (FAILED(hr)) {
            std::cerr << "Failed to map G-Buffer resource. Error: " << std::hex << hr << std::endl;
            continue;
        }

        memcpy(mappedData, meshData.data(), meshData.size() * sizeof(float));
        gBuffer->Unmap(0, nullptr);

        std::cout << "Processed and stored mesh data for one object in the G-Buffer.\n";
    }

    return true;
}

bool PointCloudEnv(ComPtr<ID3D12Device>& device, const std::vector<ComPtr<ID3D12Resource>>& sceneObjects, std::vector<float>& pointClouds)
{
    return false;
}

// CUDA Kernel for motion estimation
__global__ void MotionEstimationKernel(
    const float* prevFrame,
    const float* currFrame,
    float* motionVectors,
    int width,
    int height
) {
    // Calculate the pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we are within bounds
    if (x >= width || y >= height) return;

    int idx = y * width + x; // Flatten 2D coordinates into a 1D index

    // Compute motion vectors using absolute pixel difference
    motionVectors[idx] = fabsf(currFrame[idx] - prevFrame[idx]);
}

bool RunMotionEstimationAtMaxRate(
    const std::vector<float>& h_prevFrame,
    const std::vector<float>& h_currFrame,
    std::vector<float>& h_motionVectors,
    int width,
    int height
) {
    // Check if dimensions match
    if (h_prevFrame.size() != h_currFrame.size() || h_prevFrame.size() != h_motionVectors.size()) {
        std::cerr << "Frame sizes do not match!\n";
        return false;
    }

    // Allocate device memory
    float* d_prevFrame, * d_currFrame, * d_motionVectors;
    cudaMalloc(&d_prevFrame, width * height * sizeof(float));
    cudaMalloc(&d_currFrame, width * height * sizeof(float));
    cudaMalloc(&d_motionVectors, width * height * sizeof(float));

    // Create CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronously copy data to the device
    cudaMemcpyAsync(d_prevFrame, h_prevFrame.data(), width * height * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_currFrame, h_currFrame.data(), width * height * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Define block and grid sizes
    dim3 blockSize(16, 16); // Each block contains 16x16 threads
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // Launch the kernel asynchronously
    MotionEstimationKernel << <gridSize, blockSize, 0, stream >> > (
        d_prevFrame, d_currFrame, d_motionVectors, width, height
        );

    // Asynchronously copy results back to the host
    cudaMemcpyAsync(h_motionVectors.data(), d_motionVectors, width * height * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // Synchronize the stream to ensure all operations are complete
    cudaStreamSynchronize(stream);

    // Free the stream
    cudaStreamDestroy(stream);

    // Free device memory
    cudaFree(d_prevFrame);
    cudaFree(d_currFrame);
    cudaFree(d_motionVectors);

    return true;
}



bool EstimateMotionVectorsAndSave(
    const std::vector<float>& h_prevFrame,
    const std::vector<float>& h_currFrame,
    std::vector<float>& h_motionVectors,
    int width,
    int height
) {
    // Calculate frame size dynamically
    size_t frameSize = width * height * sizeof(float);

    // Ensure the input sizes match
    if (h_prevFrame.size() != h_currFrame.size() || h_prevFrame.size() != h_motionVectors.size()) {
        std::cerr << "Frame sizes do not match!\n";
        return false;
    }

    // Allocate device memory
    float* d_prevFrame = nullptr, * d_currFrame = nullptr, * d_motionVectors = nullptr;
    if (cudaMalloc(&d_prevFrame, frameSize) != cudaSuccess ||
        cudaMalloc(&d_currFrame, frameSize) != cudaSuccess ||
        cudaMalloc(&d_motionVectors, frameSize) != cudaSuccess) {
        std::cerr << "Failed to allocate device memory.\n";
        cudaFree(d_prevFrame);
        cudaFree(d_currFrame);
        cudaFree(d_motionVectors);
        return false;
    }

    // Copy data to the device
    if (cudaMemcpy(d_prevFrame, h_prevFrame.data(), frameSize, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_currFrame, h_currFrame.data(), frameSize, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy data to device memory.\n";
        cudaFree(d_prevFrame);
        cudaFree(d_currFrame);
        cudaFree(d_motionVectors);
        return false;
    }

    // Define block and grid sizes dynamically
    dim3 blockSize(32, 32);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // Launch the kernel
    MotionEstimationKernel << <gridSize, blockSize >> > (d_prevFrame, d_currFrame, d_motionVectors, width, height);

    // Synchronize and check for kernel execution errors
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "Kernel execution failed.\n";
        cudaFree(d_prevFrame);
        cudaFree(d_currFrame);
        cudaFree(d_motionVectors);
        return false;
    }

    // Copy results back to the host
    if (cudaMemcpy(h_motionVectors.data(), d_motionVectors, frameSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy results back to host memory.\n";
        cudaFree(d_prevFrame);
        cudaFree(d_currFrame);
        cudaFree(d_motionVectors);
        return false;
    }

    // Free device memory
    cudaFree(d_prevFrame);
    cudaFree(d_currFrame);
    cudaFree(d_motionVectors);

    return true;
}

