#ifndef CUDA_SPY_H
#define CUDA_SPY_H

#include <vector>
#include <string>
#include <windows.h>  // Include for process handling
#include "Metadata.h"
#include "PinnedMemory.h"
#include "ThreadMapper.h"

class CUDA_SPY {
public:
    CUDA_SPY(size_t bufferSize = 2024 * 2024);
    ~CUDA_SPY();

    // Initialize the monitoring system
    bool Initialize();

    // Attach to a process and map its threads
    bool AttachToProcess(const std::string& processName);

    // Monitor and log CUDA activity (updated to use hProcess)
    void MonitorCUDAActivity();

    // Replicate CUDA operations with a neural network
    bool ReplicateOperations();

    // Compare results between CUDA and CUDA_SPY
    void ComparePerformance();

    // Clean up resources
    void Shutdown();

private:
    size_t bufferSize;                 // Size of the data buffer
    std::vector<float> telemetryData;  // Data logged during CUDA activity
    Metadata metadata;                 // Store key telemetry metrics
    float* spyMemory;                  // Memory buffer for CUDA_SPY
    HANDLE hProcess;                    // **NEW: Process handle for telemetry**

    ThreadMapper threadMapper;         // Thread mapper for managing process threads
};

#endif // CUDA_SPY_H
