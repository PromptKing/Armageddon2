#include "CUDA_SPY.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "ThreadMapper.h"


CUDA_SPY::CUDA_SPY(size_t bufferSize) : bufferSize(bufferSize), spyMemory(nullptr) {
    spyMemory = PinnedMemory::Allocate<float>(bufferSize);
}

CUDA_SPY::~CUDA_SPY() {
    PinnedMemory::Free(spyMemory);
}

bool CUDA_SPY::Initialize() {
    std::cout << "CUDA_SPY initialized with buffer size: " << bufferSize << " bytes.\n";
    return true;
}

bool CUDA_SPY::AttachToProcess(const std::string& processName) {
    hProcess = threadMapper.AttachToProcess(processName); // Store the process handle
    if (!hProcess) {
        std::cerr << "[ERROR] Failed to attach to process: " << processName << ".\n";
        return false;
    }

    if (!threadMapper.MapThreads(processName)) {
        std::cerr << "[ERROR] Failed to map threads for process: " << processName << ".\n";
        return false;
    }

    std::cout << "[INFO] Successfully attached to process: " << processName << " and mapped threads.\n";
    return true;
}


void CUDA_SPY::MonitorCUDAActivity() {
    if (!hProcess) {
        std::cerr << "[ERROR] No process handle found. Ensure AttachToProcess() was called successfully.\n";
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Retrieve telemetry data from mapped threads
    int threadCount = threadMapper.GetMappedThreadCount();
    std::cout << "Monitoring " << threadCount << " threads.\n";

    telemetryData.clear(); // Reset telemetry data

    for (int i = 0; i < threadCount; ++i) {
        // **Retrieve telemetry using the process handle**
        float threadMetric = 0.0f;
        SIZE_T bytesRead;

        if (ReadProcessMemory(hProcess, (LPCVOID)(i * sizeof(float)), &threadMetric, sizeof(float), &bytesRead)) {
            telemetryData.push_back(threadMetric);
            std::cout << "[Telemetry] Thread[" << i << "] Execution Metric: " << threadMetric << "\n";
        }
        else {
            std::cerr << "[WARNING] Failed to read execution metric for Thread[" << i << "].\n";
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    metadata.executionTime = milliseconds;
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
bool CUDA_SPY::ReplicateOperations() {
    std::cout << "Replicating CUDA operations using CUDA_SPY.\n";
    return true;
}

void CUDA_SPY::ComparePerformance() {
    std::cout << "Comparing CUDA and CUDA_SPY performance.\n";
    // Example comparison logic (placeholder)
    if (telemetryData.size() > 0) {
        std::cout << "Telemetry data available for comparison.\n";
    }
    else {
        std::cerr << "No telemetry data to compare.\n";
    }
}

void CUDA_SPY::Shutdown() {
    std::cout << "CUDA_SPY shutting down and releasing resources.\n";
    PinnedMemory::Free(spyMemory);
    spyMemory = nullptr;
}
