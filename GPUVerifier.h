#pragma once
#ifndef GPUVERIFIER_H
#define GPUVERIFIER_H
#include <string>
#include <vector>

// Add this structure before the class
struct HostGPUInfo {
    std::string name;
    int deviceID;
    size_t totalMemory;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    bool supportsCUDA;
    bool supportsDirectX;
};

// GPU information structure
struct GPUInfo {
    std::string name;
    int deviceID;
    size_t totalMemory; // Total GPU memory in bytes
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    bool supportsCUDA;
    bool supportsDirectX;
};

// Class to verify and retrieve GPU information
class GPUVerifier {
public:
    GPUVerifier();
    std::vector<GPUInfo> GetAvailableGPUs() const;
    void LogGPUDetails() const;

private:
    std::vector<GPUInfo> gpuList;
    void RetrieveGPUInfo();
};

#endif // GPUVERIFIER_H
