#include <windows.h>
#include <iostream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>  // Use JSON for modern data handling
#include "cuda_runtime.h"  // Include CUDA API
#include "SupervisorData.h"
#include "CUDA_SPY.h"
#include "SupervisorData.cpp"

#define PIPE_NAME "\\\\.\\pipe\\ArmageddonMetrics"


// Function to update CUDA_SPY with new metrics
void updateCudaSpy(float system_cpu, float process_cpu, float gpu_usage) {
    cudaError_t err;

    // Example CUDA kernel launch (dummy example)
    err = cudaSetDevice(0); // Assuming GPU 0
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    // TODO: Implement actual CUDA kernel logic here based on real-time performance data
    std::cout << "[CUDA_SPY] Updating GPU workload: System CPU: " << system_cpu
        << "%, Process CPU: " << process_cpu << "%, GPU Usage: " << gpu_usage << "%" << std::endl;
}

// Function to send data to SupervisorData
void updateSupervisorData(float system_cpu, float process_cpu, float gpu_usage, float memory_usage) {
    SupervisorData supervisor;
    supervisor.setCpuUsage(system_cpu, process_cpu);
    supervisor.setGpuUsage(gpu_usage);
    supervisor.setMemoryUsage(memory_usage);

    // Log the received data
    std::cout << "[SupervisorData] Metrics Updated: System CPU: " << system_cpu
        << "%, Process CPU: " << process_cpu << "%, GPU Usage: " << gpu_usage
        << "%, Memory Usage: " << memory_usage << "MB" << std::endl;
}

// Function to read metrics from Named Pipe and process them
void ReadMetricsFromPipe() {
    HANDLE hPipe;
    char buffer[65536];
    DWORD bytesRead;

    // Try connecting to the Named Pipe
    hPipe = CreateFileA(PIPE_NAME, GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);

    if (hPipe == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to connect to Named Pipe." << std::endl;
        return;
    }

    std::cout << "Connected to Named Pipe. Receiving metrics..." << std::endl;

    while (true) {
        if (ReadFile(hPipe, buffer, sizeof(buffer) - 1, &bytesRead, NULL)) {
            buffer[bytesRead] = '\0';  // Null-terminate string

            // Convert JSON string to structured data
            std::string jsonString(buffer);
            auto metrics = nlohmann::json::parse(jsonString);

            // Extract performance data
            float system_cpu = metrics["system_cpu"];
            float process_cpu = metrics["process_cpu"];
            float gpu_usage = metrics["gpu_usage"];
            float memory_usage = metrics["memory_rss_mb"];

            // Send the received data to CUDA_SPY
            updateCudaSpy(system_cpu, process_cpu, gpu_usage);

            // Send the received data to SupervisorData
            updateSupervisorData(system_cpu, process_cpu, gpu_usage, memory_usage);
        }

        Sleep(1000);  // Read every second
    }

    CloseHandle(hPipe);
}

int main() {
    ReadMetricsFromPipe();
    return 0;
}

