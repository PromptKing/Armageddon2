#ifndef CUDA_D3D12_DEVICE_H
#define CUDA_D3D12_DEVICE_H

#include <windows.h>
#include <cuda_runtime.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <vector>
#include <npp.h>
#include <DirectXMath.h>

using Microsoft::WRL::ComPtr;

/**
 * @brief Compiles an HLSL shader from a file.
 *
 * @param filepath Path to the shader file.
 * @param entryPoint Shader entry point function.
 * @param target Shader model target (e.g., "vs_5_0").
 * @return Compiled shader blob or nullptr if compilation fails.
 */
ComPtr<ID3DBlob> CompileShader(const std::wstring& filepath, const char* entryPoint, const char* target);

/**
 * @brief Initializes the tessellation pipeline.
 *
 * @param device Direct3D 12 device.
 * @param pipelineState Created pipeline state.
 * @param rootSignature Created root signature.
 * @return True if successful, false otherwise.
 */
bool InitializeTessellationPipeline(
    ComPtr<ID3D12Device>& device,
    ComPtr<ID3D12PipelineState>& pipelineState,
    ComPtr<ID3D12RootSignature>& rootSignature
);

/**
 * @brief Initializes Direct3D 12 and CUDA interop.
 *
 * @param device Direct3D 12 device.
 * @param resource Direct3D12 resource for interop.
 * @param width Frame width.
 * @param height Frame height.
 * @return True if successful, false otherwise.
 */
bool InitializeD3D12AndCUDA(
    ComPtr<ID3D12Device>& device,
    ComPtr<ID3D12Resource>& resource,
    UINT width,
    UINT height
);

/**
 * @brief Overrides 3D object properties by updating the vertex buffer.
 *
 * @param resource Direct3D12 resource to be updated.
 * @param vertexData New vertex data.
 * @return True if successful, false otherwise.
 */
bool Override3DObjectProperties(
    ComPtr<ID3D12Resource>& resource,
    const std::vector<float>& vertexData
);

/**
 * @brief Generates a real-time frame from motion vectors.
 *
 * @param resource Direct3D12 resource to store the frame.
 * @param previousFrameData Data from the previous frame.
 * @param nextFrameData Data from the next frame.
 * @return True if successful, false otherwise.
 */
bool GenerateRealTimeFrameFromMotionVectors(
    ComPtr<ID3D12Resource>& resource,
    const std::vector<float>& previousFrameData,
    const std::vector<float>& nextFrameData
);

/**
 * @brief Predicts and displays the next frame based on motion vectors.
 *
 * @param resource Direct3D12 resource to display the frame.
 * @param motionVectors Calculated motion vectors for frame prediction.
 * @return True if successful, false otherwise.
 */
bool PredictAndDisplayNextFrame(
    ComPtr<ID3D12Resource>& resource,
    const std::vector<float>& motionVectors
);

/**
 * @brief Captures and calculates 3D meshes, applies dynamic weights, and stores them in the G-Buffer.
 *
 * @param device Direct3D 12 device.
 * @param gBuffer G-Buffer resource for storing mesh and texture data.
 * @param sceneObjects Vector of scene object resources to process.
 * @return True if successful, false otherwise.
 */
bool MeshCalculationAndReconstruction(
    ComPtr<ID3D12Device>& device,
    ComPtr<ID3D12Resource>& gBuffer,
    const std::vector<ComPtr<ID3D12Resource>>& sceneObjects
);

/**
 * @brief Generates a point cloud environment.
 *
 * @param device Direct3D 12 device.
 * @param sceneObjects Vector of scene object resources.
 * @param pointClouds Output vector for point cloud data.
 * @return True if successful, false otherwise.
 */
bool PointCloudEnv(
    ComPtr<ID3D12Device>& device,
    const std::vector<ComPtr<ID3D12Resource>>& sceneObjects,
    std::vector<float>& pointClouds
);

/**
 * @brief Estimates motion vectors between two frames.
 *
 * @param prevFrame Resource containing the previous frame.
 * @param currFrame Resource containing the current frame.
 * @param motionVectors Output motion vectors resource.
 * @return True if successful, false otherwise.
 */
bool EstimateMotionVectors(
    ComPtr<ID3D12Resource>& prevFrame,
    ComPtr<ID3D12Resource>& currFrame,
    ComPtr<ID3D12Resource>& motionVectors
);

/**
 * @brief Lighting structure definition.
 */
struct Lighting {
    DirectX::XMFLOAT3 lightDirection; // Direction of the light
    float padding;                   // Padding for alignment
    DirectX::XMFLOAT4 lightColor;    // Light color (RGBA)
    DirectX::XMFLOAT4 ambientColor;  // Ambient light color
};

/**
 * @brief Generates dynamic weights for meshes or textures.
 *
 * @param size Number of elements.
 * @param baseWeight Base weight value.
 * @param variance Weight variance.
 * @return Vector of dynamically generated weights.
 */
std::vector<float> GenerateDynamicWeights(size_t size, float baseWeight, float variance);

// Include Dual GPU RTV Helper
#ifndef DUAL_GPU_RTV_HELPER_H
#define DUAL_GPU_RTV_HELPER_H


#include "d3dx12.h"
#include <wrl.h>
#include <stdexcept>
#include <iostream>

using Microsoft::WRL::ComPtr;

// Function to create a Render Target View (RTV) Descriptor Heap
inline ComPtr<ID3D12DescriptorHeap> CreateRTVHeap(
    ID3D12Device* device,
    UINT numDescriptors
) {
    if (!device) {
        throw std::invalid_argument("Invalid device pointer");
    }

    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.NumDescriptors = numDescriptors;
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    HRESULT hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&rtvHeap));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create RTV heap");
    }

    return rtvHeap;
}

// Function to create a Command Queue
inline ComPtr<ID3D12CommandQueue> CreateCommandQueue(
    ID3D12Device* device,
    D3D12_COMMAND_LIST_TYPE type
) {
    if (!device) {
        throw std::invalid_argument("Invalid device pointer");
    }

    D3D12_COMMAND_QUEUE_DESC desc = {};
    desc.Type = type;
    desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    desc.NodeMask = 0;

    ComPtr<ID3D12CommandQueue> commandQueue;
    HRESULT hr = device->CreateCommandQueue(&desc, IID_PPV_ARGS(&commandQueue));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create command queue");
    }

    return commandQueue;
}

// Function to create a Fence
inline ComPtr<ID3D12Fence> CreateFence(
    ID3D12Device* device,
    UINT64 initialValue
) {
    if (!device) {
        throw std::invalid_argument("Invalid device pointer");
    }

    ComPtr<ID3D12Fence> fence;
    HRESULT hr = device->CreateFence(initialValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create fence");
    }

    return fence;
}

// Function to signal the GPU
inline void SignalGPU(
    ID3D12CommandQueue* commandQueue,
    ID3D12Fence* fence,
    UINT64& fenceValue
) {
    if (!commandQueue || !fence) {
        throw std::invalid_argument("Invalid command queue or fence pointer");
    }

    HRESULT hr = commandQueue->Signal(fence, fenceValue++);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to signal GPU");
    }
}

// Function to wait for the GPU
inline void WaitForGPU(
    ID3D12CommandQueue* commandQueue,
    ID3D12Fence* fence,
    UINT64 fenceValue
) {
    if (!commandQueue || !fence) {
        throw std::invalid_argument("Invalid command queue or fence pointer");
    }

    HRESULT hr = commandQueue->Wait(fence, fenceValue);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to wait for GPU");
    }
}

// Function to transition a resource between states
inline void TransitionResource(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES stateBefore,
    D3D12_RESOURCE_STATES stateAfter
) {
    if (!commandList || !resource) {
        throw std::invalid_argument("Invalid command list or resource pointer");
    }

   commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        resource,
        stateBefore,
        stateAfter
    ));
}

// Function to create and initialize fence values
inline void InitializeFenceValues(UINT64& fenceValue) {
    fenceValue = 1; // Initial value for the fence
}

// Function to set render targets
inline void SetRenderTarget(
    ID3D12GraphicsCommandList* commandList,
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle
) {
    if (!commandList) {
        throw std::invalid_argument("Invalid command list pointer");
    }

    commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
}

#endif // DUAL_GPU_RTV_HELPER_H

#endif // CUDA_D3D12_DEVICE_H
