#include "ThreadMapper.h"
#include <TlHelp32.h>
#include <cuda_runtime.h>
#include <iostream>

ThreadMapper::ThreadMapper() : d_gpuBuffer(nullptr), bufferSize(0) {}

ThreadMapper::~ThreadMapper() {
    if (d_gpuBuffer) {
        cudaFree(d_gpuBuffer);
        d_gpuBuffer = nullptr;
    }
}

// Attach to the process by name
HANDLE ThreadMapper::AttachToProcess(const std::string& processName) {
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: Failed to create process snapshot." << std::endl;
        return nullptr;
    }

    PROCESSENTRY32 pe32 = { sizeof(PROCESSENTRY32) };
    if (Process32First(hSnapshot, &pe32)) {
        do {
#ifdef UNICODE
            std::wstring exeName(pe32.szExeFile);
            std::string exeNameStr = std::string(exeName.begin(), exeName.end());
#else
            std::string exeNameStr = pe32.szExeFile;
#endif
            if (exeNameStr == processName) {
                CloseHandle(hSnapshot);
                HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pe32.th32ProcessID);
                if (!hProcess) {
                    std::cerr << "Error: Failed to open process. Error code: " << GetLastError() << std::endl;
                }
                return hProcess;
            }
        } while (Process32Next(hSnapshot, &pe32));
    }

    std::cerr << "Error: Process " << processName << " not found." << std::endl;
    CloseHandle(hSnapshot);
    return nullptr;
}

// Map threads in the target application
bool ThreadMapper::MapThreads(const std::string& processName) {
    HANDLE hProcess = AttachToProcess(processName);
    if (!hProcess) {
        return false;
    }

    HANDLE hThreadSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (hThreadSnapshot == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: Failed to create thread snapshot." << std::endl;
        CloseHandle(hProcess);
        return false;
    }

    THREADENTRY32 te32 = { sizeof(THREADENTRY32) };
    if (Thread32First(hThreadSnapshot, &te32)) {
        do {
            if (te32.th32OwnerProcessID == GetProcessId(hProcess)) {
                ThreadInfo threadInfo;
                threadInfo.threadID = te32.th32ThreadID;

                if (RetrieveThreadData(hProcess, threadInfo)) {
                    std::lock_guard<std::mutex> lock(threadMutex);
                    mappedThreads.push_back(threadInfo);
                }
            }
        } while (Thread32Next(hThreadSnapshot, &te32));
    }
    else {
        std::cerr << "Error: Could not retrieve threads for process." << std::endl;
    }

    CloseHandle(hThreadSnapshot);
    CloseHandle(hProcess);

    std::cout << "Mapped " << mappedThreads.size() << " threads in process: " << processName << std::endl;
    return !mappedThreads.empty();
}

// Retrieve data from the thread
bool ThreadMapper::RetrieveThreadData(HANDLE hProcess, ThreadInfo& threadInfo) {
    // Placeholder: Implement logic to extract thread start address
    threadInfo.threadStartAddress = nullptr;

    // Simulate copying function data
    threadInfo.cpuFunctionData = { 0x48, 0x89, 0xE5, 0xC3 }; // Example machine code
    return true;
}

// Transfer CPU functions to GPU
bool ThreadMapper::TransferCPUFunctionsToGPU() {
    if (mappedThreads.empty()) {
        std::cerr << "Error: No threads mapped to transfer to GPU." << std::endl;
        return false;
    }

    // Calculate the buffer size required
    bufferSize = 0;
    for (const auto& thread : mappedThreads) {
        bufferSize += thread.cpuFunctionData.size();
    }

    // Allocate GPU memory
    cudaError_t cudaStatus = cudaMalloc(&d_gpuBuffer, bufferSize);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error: Failed to allocate GPU buffer. CUDA Error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return false;
    }

    // Copy thread data to GPU buffer
    size_t offset = 0;
    for (const auto& thread : mappedThreads) {
        cudaStatus = cudaMemcpy(d_gpuBuffer + offset, thread.cpuFunctionData.data(),
            thread.cpuFunctionData.size(), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Error: Failed to copy CPU function data to GPU. CUDA Error: " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(d_gpuBuffer);
            d_gpuBuffer = nullptr;
            return false;
        }
        offset += thread.cpuFunctionData.size();
    }

    std::cout << "Successfully transferred CPU functions to GPU buffer." << std::endl;
    return true;
}

// Accessor for GPU buffer
unsigned char* ThreadMapper::GetGPUBuffer() const {
    return d_gpuBuffer;
}

// Accessor for GPU buffer size
size_t ThreadMapper::GetGPUBufferSize() const {
    return bufferSize;
}

// Accessor for mapped thread count
int ThreadMapper::GetMappedThreadCount() const {
    return static_cast<int>(mappedThreads.size());
}
