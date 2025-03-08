#ifndef DUAL_GPU_RTV_HELPER_H
#define DUAL_GPU_RTV_HELPER_H

#include <d3d12.h>
#include "d3dx12.h"
#include <dxgi1_6.h>
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
