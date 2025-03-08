#include "ScreenCapture.h"
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dx12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <iostream>

using Microsoft::WRL::ComPtr;

// Global variables for D3D11 and D3D12
enum class GraphicsAPI { None, Direct3D11, Direct3D12 };
GraphicsAPI g_currentAPI = GraphicsAPI::None;

ComPtr<ID3D11Device> g_d3d11Device;
ComPtr<ID3D11DeviceContext> g_d3d11Context;
ComPtr<IDXGISwapChain> g_d3d11SwapChain;

ComPtr<ID3D12Device> g_d3d12Device;
ComPtr<ID3D12CommandQueue> g_d3d12CommandQueue;
ComPtr<IDXGISwapChain3> g_d3d12SwapChain;

// Detect and initialize the appropriate DirectX version
bool InitializeGraphicsAPI() {
    // Try initializing Direct3D 12 first
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_d3d12Device));
    if (SUCCEEDED(hr)) {
        std::cout << "Direct3D 12 initialized successfully." << std::endl;
        g_currentAPI = GraphicsAPI::Direct3D12;
        return true;
    }

    // If Direct3D 12 is not available, fallback to Direct3D 11
    hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &g_d3d11Device,
        nullptr,
        &g_d3d11Context
    );

    if (SUCCEEDED(hr)) {
        std::cout << "Direct3D 11 initialized successfully." << std::endl;
        g_currentAPI = GraphicsAPI::Direct3D11;
        return true;
    }

    std::cerr << "Failed to initialize Direct3D 11 or 12." << std::endl;
    g_currentAPI = GraphicsAPI::None;
    return false;
}

bool GetScreenDataD3D11(float* buffer, int width, int height) {
    if (!buffer || width <= 0 || height <= 0) {
        std::cerr << "Invalid buffer or dimensions." << std::endl;
        return false;
    }

    // Step 1: Initialize Direct3D 11
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain> swapChain;

    // Step 2: Get the back buffer
    ComPtr<ID3D11Texture2D> backBuffer;
    HRESULT hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(backBuffer.GetAddressOf()));
    if (FAILED(hr)) {
        std::cerr << "Failed to get swap chain back buffer: " << std::hex << hr << std::endl;
        return false;
    }

    // Step 3: Create a staging texture for CPU read access
    D3D11_TEXTURE2D_DESC desc = {};
    backBuffer->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.BindFlags = 0;

    ComPtr<ID3D11Texture2D> stagingTexture;
    hr = device->CreateTexture2D(&desc, nullptr, stagingTexture.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: " << std::hex << hr << std::endl;
        return false;
    }

    // Step 4: Copy the back buffer to the staging texture
    context->CopyResource(stagingTexture.Get(), backBuffer.Get());

    // Step 5: Map the staging texture for CPU read access
    D3D11_MAPPED_SUBRESOURCE mappedResource = {};
    hr = context->Map(stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mappedResource);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture: " << std::hex << hr << std::endl;
        return false;
    }

    // Step 6: Convert pixel data and store it in the buffer
    const BYTE* srcData = static_cast<const BYTE*>(mappedResource.pData);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int dstIdx = (y * width + x) * 3;  // RGB channels
            int srcIdx = y * mappedResource.RowPitch + x * 4; // RGBA format

            buffer[dstIdx + 0] = srcData[srcIdx + 2] / 255.0f; // R
            buffer[dstIdx + 1] = srcData[srcIdx + 1] / 255.0f; // G
            buffer[dstIdx + 2] = srcData[srcIdx + 0] / 255.0f; // B
        }
    }

    // Step 7: Unmap the resource
    context->Unmap(stagingTexture.Get(), 0);

    std::cout << "Captured screen successfully using Direct3D 11." << std::endl;
    return true;
}


// Direct3D 11 implementation
bool InitializeD3D11(ID3D11Device** device, ID3D11DeviceContext** context, IDXGISwapChain** swapChain, int width, int height) {
    DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
    swapChainDesc.BufferCount = 1;
    swapChainDesc.BufferDesc.Width = width;
    swapChainDesc.BufferDesc.Height = height;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = GetDesktopWindow();
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.Windowed = TRUE;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &swapChainDesc,
        swapChain,
        device,
        nullptr,
        context
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to initialize Direct3D 11: " << std::hex << hr << std::endl;
        return false;
    }

    return true;
}

bool GetScreenData(float* buffer, int width, int height) {
    if (!buffer || width <= 0 || height <= 0) {
        std::cerr << "Invalid buffer or dimensions." << std::endl;
        return false;
    }

    // Initialize Direct3D 11
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain> swapChain;

    if (!InitializeD3D11(device.GetAddressOf(), context.GetAddressOf(), swapChain.GetAddressOf(), width, height)) {
        return false;
    }

    // Create a staging texture to read screen data
    ComPtr<ID3D11Texture2D> backBuffer;
    swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(backBuffer.GetAddressOf()));

    D3D11_TEXTURE2D_DESC desc = {};
    backBuffer->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.BindFlags = 0;

    ComPtr<ID3D11Texture2D> stagingTexture;
    HRESULT hr = device->CreateTexture2D(&desc, nullptr, stagingTexture.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: " << std::hex << hr << std::endl;
        return false;
    }

    // Copy the back buffer to the staging texture
    context->CopyResource(stagingTexture.Get(), backBuffer.Get());

    // Map the staging texture to read the pixel data
    D3D11_MAPPED_SUBRESOURCE mappedResource = {};
    hr = context->Map(stagingTexture.Get(), 0, D3D11_MAP_READ, 0, &mappedResource);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture: " << std::hex << hr << std::endl;
        return false;
    }

    // Copy the pixel data to the buffer
    const BYTE* srcData = static_cast<const BYTE*>(mappedResource.pData);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int dstIdx = (y * width + x) * 3;
            int srcIdx = y * mappedResource.RowPitch + x * 4; // RGBA

            buffer[dstIdx + 0] = srcData[srcIdx + 2] / 255.0f; // R
            buffer[dstIdx + 1] = srcData[srcIdx + 1] / 255.0f; // G
            buffer[dstIdx + 2] = srcData[srcIdx + 0] / 255.0f; // B
        }
    }

    context->Unmap(stagingTexture.Get(), 0);

    return true;
}

// Direct3D 12 implementation
bool InitializeD3D12(
    ComPtr<ID3D12Device>& device,
    ComPtr<ID3D12CommandQueue>& commandQueue,
    ComPtr<IDXGISwapChain3>& swapChain,
    ComPtr<ID3D12DescriptorHeap>& rtvHeap,
    ComPtr<ID3D12Resource>& backBuffer,
    int width, int height
) {
    // Create a DXGI factory
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        std::cerr << "Failed to create DXGI factory: " << std::hex << hr << std::endl;
        return false;
    }

    // Create a Direct3D 12 device
    hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D12 device: " << std::hex << hr << std::endl;
        return false;
    }

    // Create a command queue
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(&commandQueue));
    if (FAILED(hr)) {
        std::cerr << "Failed to create command queue: " << std::hex << hr << std::endl;
        return false;
    }

    // Create a swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.Width = width;
    swapChainDesc.Height = height;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = 2;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    ComPtr<IDXGISwapChain1> tempSwapChain;
    hr = factory->CreateSwapChainForHwnd(
        commandQueue.Get(), GetDesktopWindow(), &swapChainDesc, nullptr, nullptr, &tempSwapChain
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create swap chain: " << std::hex << hr << std::endl;
        return false;
    }

    hr = tempSwapChain.As(&swapChain);
    if (FAILED(hr)) {
        std::cerr << "Failed to cast swap chain: " << std::hex << hr << std::endl;
        return false;
    }

    // Create RTV descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = 2;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    hr = device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap));
    if (FAILED(hr)) {
        std::cerr << "Failed to create RTV descriptor heap: " << std::hex << hr << std::endl;
        return false;
    }

    // Get the back buffer
    hr = swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    if (FAILED(hr)) {
        std::cerr << "Failed to get back buffer: " << std::hex << hr << std::endl;
        return false;
    }

    return true;
}

bool GetScreenDataD3D12(float* buffer, int width, int height) {
    if (!buffer || width <= 0 || height <= 0) {
        std::cerr << "Invalid buffer or dimensions." << std::endl;
        return false;
    }

    // Initialize Direct3D 12
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> commandQueue;
    ComPtr<IDXGISwapChain3> swapChain;
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12Resource> backBuffer;

    if (!InitializeD3D12(device, commandQueue, swapChain, rtvHeap, backBuffer, width, height)) {
        return false;
    }

    // Map the back buffer to retrieve pixel data
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    D3D12_RESOURCE_DESC desc = backBuffer->GetDesc();
    footprint.Footprint.Width = static_cast<UINT>(desc.Width);
    footprint.Footprint.Height = static_cast<UINT>(desc.Height);
    footprint.Footprint.Depth = 1;
    footprint.Footprint.RowPitch = (desc.Width * 4 + 255) & ~255; // Align to 256 bytes
    footprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

    ComPtr<ID3D12Resource> stagingBuffer;
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC stagingDesc = {};
    stagingDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    stagingDesc.Width = footprint.Footprint.RowPitch * desc.Height;
    stagingDesc.Height = 1;
    stagingDesc.DepthOrArraySize = 1;
    stagingDesc.MipLevels = 1;
    stagingDesc.Format = DXGI_FORMAT_UNKNOWN;
    stagingDesc.SampleDesc.Count = 1;
    stagingDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    HRESULT hr = device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &stagingDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&stagingBuffer)
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging buffer: " << std::hex << hr << std::endl;
        return false;
    }

    // Copy back buffer data to the staging buffer
    // (Set up command lists, execute, and read back data)

    // Convert pixel data to the buffer (logic similar to D3D11 version)

    return true;
}

