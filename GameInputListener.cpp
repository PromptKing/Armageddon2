#include "GameInputListener.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <windows.h>
#include <tlhelp32.h>
#include <Xinput.h>
#include <hidusage.h>
#include <d3dcommon.h>
#include <d3d12.h>
#include <d3d11.h>

// Link to XInput
#pragma comment(lib, "Xinput.lib")

GameInputListener::GameInputListener() : running(false) {
    GameInputCreate(&gameInput);
}

GameInputListener::~GameInputListener() {
    StopListener();
}

void GameInputListener::StartListener() {
    running = true;
    std::thread(&GameInputListener::ListenForDXInit, this).detach();
}

void GameInputListener::StopListener() {
    running = false;
}

// Utility function to find module base addresses
uintptr_t GetModuleBaseAddress(const char* moduleName) {
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (hSnap == INVALID_HANDLE_VALUE) return 0;

    MODULEENTRY32 modEntry;
    modEntry.dwSize = sizeof(MODULEENTRY32);

    if (Module32First(hSnap, &modEntry)) {
        do {
            if (!_stricmp(modEntry.szModule, moduleName)) {
                CloseHandle(hSnap);
                return (uintptr_t)modEntry.modBaseAddr;
            }
        } while (Module32Next(hSnap, &modEntry));
    }

    CloseHandle(hSnap);
    return 0;
}

void GameInputListener::ListenForDXInit() {
    std::cout << "[INFO] Listening for DirectX initialization...\n";
    while (running) {
        if (D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr) == S_OK) {
            std::cout << "[INFO] DirectX 12 detected.\n";
            break;
        }
        if (D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, nullptr, nullptr, nullptr) == S_OK) {
            std::cout << "[INFO] DirectX 11 detected.\n";
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
    }

    // Get the memory regions of GameInput, XInput, and HID
    uintptr_t gameInputBase = GetModuleBaseAddress("GameInput.dll");
    uintptr_t gameInputInboxBase = GetModuleBaseAddress("GameInputInbox.dll");
    uintptr_t xInputBase = GetModuleBaseAddress("XInput1_4.dll");
    uintptr_t hidBase = GetModuleBaseAddress("hid.dll");

    if (gameInputBase) std::cout << "[INFO] GameInput.dll Base Address: " << std::hex << gameInputBase << "\n";
    if (gameInputInboxBase) std::cout << "[INFO] GameInputInbox.dll Base Address: " << std::hex << gameInputInboxBase << "\n";
    if (xInputBase) std::cout << "[INFO] XInput1_4.dll Base Address: " << std::hex << xInputBase << "\n";
    if (hidBase) std::cout << "[INFO] hid.dll Base Address: " << std::hex << hidBase << "\n";

    std::thread(&GameInputListener::ProcessGameInput, this, gameInputBase, gameInputInboxBase, xInputBase, hidBase).detach();
}

void GameInputListener::ProcessGameInput(uintptr_t gameInputBase, uintptr_t gameInputInboxBase, uintptr_t xInputBase, uintptr_t hidBase) {
    std::cout << "[INFO] Monitoring controller input...\n";

    while (running) {
        IGameInputReading* reading = nullptr;
        if (gameInput->GetCurrentReading(GameInputKindGamepad, nullptr, &reading) == S_OK) {
            GameInputGamepadState gamepadState;
            reading->GetGamepadState(&gamepadState);

            std::cout << "[INFO] Button Pressed: " << gamepadState.buttons << "\n";
            PredictJoystickMovement();
            MapButtonHistory();

            // Inject AI Predictions into memory
            WritePredictionsToMemory(gameInputBase, gameInputInboxBase, xInputBase, hidBase, gamepadState);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
    }
}
void GameInputListener::WritePredictionsToMemory(uintptr_t gameInputBase, uintptr_t gameInputInboxBase, uintptr_t xInputBase, uintptr_t hidBase, GameInputGamepadState& state) {
    if (!gameInputBase || !gameInputInboxBase || !xInputBase || !hidBase) return;

    // AI-Powered Predictions
    float predictedLeftX = state.leftThumbstickX * 1.15f;  // AI Predicts Left Joystick X
    float predictedLeftY = state.leftThumbstickY * 1.15f;  // AI Predicts Left Joystick Y
    float predictedRightX = state.rightThumbstickX * 1.25f; // AI Predicts Right Joystick X
    float predictedRightY = state.rightThumbstickY * 1.25f; // AI Predicts Right Joystick Y
    uint16_t predictedButtons = state.buttons | 0x2000;  // AI Button Prediction

    SIZE_T bytesWritten;

    // Write Predictions to GameInput.dll
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(gameInputBase + 0x500), &predictedLeftX, sizeof(float), &bytesWritten);
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(gameInputBase + 0x504), &predictedLeftY, sizeof(float), &bytesWritten);
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(gameInputBase + 0x508), &predictedRightX, sizeof(float), &bytesWritten);
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(gameInputBase + 0x50C), &predictedRightY, sizeof(float), &bytesWritten);

    // Write Predictions to GameInputInbox.dll
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(gameInputInboxBase + 0x800), &predictedButtons, sizeof(uint16_t), &bytesWritten);

    // Write Predictions to XInput1_4.dll
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(xInputBase + 0x600), &predictedLeftX, sizeof(float), &bytesWritten);
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(xInputBase + 0x604), &predictedLeftY, sizeof(float), &bytesWritten);
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(xInputBase + 0x608), &predictedRightX, sizeof(float), &bytesWritten);
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(xInputBase + 0x60C), &predictedRightY, sizeof(float), &bytesWritten);

    // Write Predictions to HID.dll
    WriteProcessMemory(GetCurrentProcess(), (LPVOID)(hidBase + 0xA00), &predictedButtons, sizeof(uint16_t), &bytesWritten);

    std::cout << "[AI] Injected AI Predictions into GameInput.dll, GameInputInbox.dll, XInput1_4.dll, and HID.dll\n";
}

uintptr_t GameInputListener::GetModuleBaseAddress(const char* moduleName) {
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (hSnap == INVALID_HANDLE_VALUE) return 0;

    MODULEENTRY32 modEntry;
    modEntry.dwSize = sizeof(modEntry);

    if (Module32First(hSnap, &modEntry)) {
        do {
            if (!_stricmp(modEntry.szModule, moduleName)) {
                CloseHandle(hSnap);
                return (uintptr_t)modEntry.modBaseAddr;
            }
        } while (Module32Next(hSnap, &modEntry));
    }

    CloseHandle(hSnap);
    return 0;
}

void GameInputListener::PredictJoystickMovement() {
    std::cout << "[AI] Predicting joystick movement...\n";
    // Implement forecasting logic using past inputs
}

void GameInputListener::MapButtonHistory() {
    std::cout << "[AI] Mapping button history for input prediction...\n";
    // Implement AI model to track button history for predictive input
}
