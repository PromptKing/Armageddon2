#include "DirectXLinker.h"
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <TlHelp32.h>
#include <psapi.h> // For EnumProcessModules and GetModuleBaseName

// Constructor
DirectXLinker::DirectXLinker() {}

// Destructor
DirectXLinker::~DirectXLinker() {}

// Initialize the DirectX Linker
bool DirectXLinker::Initialize() {
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) {
        std::cerr << "Failed to create DXGI Factory. Error: " << std::hex << hr << std::endl;
        return false;
    }
    std::cout << "DirectXLinker initialized successfully.\n";
    return true;
}

// Search for DirectX pipelines in all running applications
bool DirectXLinker::SearchAndInject() {
    // Enumerate all processes
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to create process snapshot.\n";
        return false;
    }

    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);

    if (Process32First(hSnapshot, &pe32)) {
        do {
            // Check if the process is using DirectX
            if (IsUsingDirectX(pe32.th32ProcessID)) {
                std::wcout << L"Found application using DirectX: " << pe32.szExeFile << L" (PID: " << pe32.th32ProcessID << L")\n";

                // Inject into the application's pipeline
                HookPipeline(pe32.th32ProcessID);
            }
        } while (Process32Next(hSnapshot, &pe32));
    }
    else {
        std::cerr << "Failed to enumerate processes.\n";
    }

    CloseHandle(hSnapshot);
    return true;
}

// Check if the process is using DirectX
bool DirectXLinker::IsUsingDirectX(DWORD processID) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processID);
    if (!hProcess) {
        return false; // Cannot open process
    }

    HMODULE hMods[1024];
    DWORD cbNeeded;

    // Enumerate the modules loaded by the process
    if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &cbNeeded)) {
        for (unsigned int i = 0; i < (cbNeeded / sizeof(HMODULE)); i++) {
#ifdef UNICODE
            wchar_t szModName[MAX_PATH];
            if (GetModuleBaseNameW(hProcess, hMods[i], szModName, MAX_PATH)) {
                std::wstring moduleName(szModName);

                if (moduleName.find(L"d3dcompiler_47.dll") != std::wstring::npos ||
                    moduleName.find(L"BeamNGBlendLayerDSP.dll") != std::wstring::npos ||
                    moduleName.find(L"dxcompiler.dll") != std::wstring::npos) {
                    CloseHandle(hProcess);
                    return true; // DirectX module found
                }
            }
#else
            char szModName[MAX_PATH];
            if (GetModuleBaseNameA(hProcess, hMods[i], szModName, MAX_PATH)) {
                std::string moduleName(szModName);

                if (moduleName.find("d3dcompiler_47.dll") != std::string::npos ||
                    moduleName.find("BeamNGBlendLayerDSP.dll") != std::string::npos ||
                    moduleName.find("draco.dll") != std::string::npos ||
                    moduleName.find("Armorrgeddon Hyper Physics Calulations.dll") != std::string::npos ||
                    moduleName.find("cublas64_100.dll") != std::string::npos ||
                    moduleName.find("cudart64_100.dll") != std::string::npos ||
                    moduleName.find("curand64_100.dll") != std::string::npos ||
                    moduleName.find("cusparse64_100.dll") != std::string::npos ||
                    moduleName.find("Ultron.dll") != std::string::npos ||
                    moduleName.find("nvml.dll") != std::string::npos ||
                    moduleName.find("D3D12Core.dll") != std::string::npos ||
                    moduleName.find("D3D12SDKLayers.dll") != std::string::npos ||
                    moduleName.find("d3dcsx_46.dll") != std::string::npos ||
                    moduleName.find("F:\\Misc\\Unreal Engine\\UE_5.5\\Engine\\Binaries\\Win64") != std::string::npos ||
                    moduleName.find("GFSDK_ShadowLib.win64.dll") != std::string::npos) {
                    CloseHandle(hProcess);
                    return true; // DirectX module found
                }
            }
#endif
        }
    }

    CloseHandle(hProcess);
    return false;
}

// Hook into the application's pipeline
bool DirectXLinker::HookPipeline(DWORD processID) {
    // Placeholder for hooking logic
    std::cout << "Injecting into process (PID: " << processID << ")...\n";

    // Extend this section with actual hooking code for swap chains or command queues
    return true;
}
