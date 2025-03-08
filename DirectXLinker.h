#ifndef DIRECTX_LINKER_H
#define DIRECTX_LINKER_H

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <iostream>

using Microsoft::WRL::ComPtr;

/**
 * @brief DirectX_2_DirectX Linker (DDL) class.
 * Responsible for enumerating and injecting into existing DirectX pipelines.
 */
class DirectXLinker {
public:
    DirectXLinker();
    ~DirectXLinker();

    /**
     * @brief Initializes the DirectX Linker.
     *
     * @return True if successful, false otherwise.
     */
    bool Initialize();

    /**
     * @brief Searches for active DirectX pipelines in all running applications.
     *
     * @return True if pipelines are found and linked successfully, false otherwise.
     */
    bool SearchAndInject();

private:
    // Check if a process is using DirectX
    bool IsUsingDirectX(DWORD processID);

    // Hook into a process's DirectX pipeline
    bool HookPipeline(DWORD processID);

    // DXGI Factory for enumerating adapters
    ComPtr<IDXGIFactory6> dxgiFactory;
};

#endif // DIRECTX_LINKER_H
