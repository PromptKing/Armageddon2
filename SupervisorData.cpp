#include "SupervisorData.h"
#include <iostream>
#include <vector>
#include "ThreadMapper.h"


// Initialize SupervisorData for a given number of threads
void ThreadMapper(SupervisorData* supervisorData, int numThreads) {
    for (int i = 0; i < numThreads; ++i) {
        supervisorData[i].threadID = i;
        supervisorData[i].executionTime = 0.0f;
        supervisorData[i].prediction = 0.0f;
        supervisorData[i].memoryRegion = nullptr;
        supervisorData[i].isHalted = false;
        supervisorData[i].matrixData.clear(); // Ensure matrix data is empty
        supervisorData[i].cpuComputationResult = 0.5f;
        supervisorData[i].gpuComputationResult = 0.5f;
    }
}

// Update SupervisorData after matrix multiplication
void UpdateMatrixData(SupervisorData* supervisorData, int threadID, const std::vector<float>& matrixResult) {
    supervisorData[threadID].matrixData = matrixResult;
}

// Update SupervisorData after CPU computation
void UpdateCPUComputation(SupervisorData* supervisorData, int threadID, float result) {
    supervisorData[threadID].cpuComputationResult = result;
}

// Update SupervisorData after GPU computation
void UpdateGPUComputation(SupervisorData* supervisorData, int threadID, float result) {
    supervisorData[threadID].gpuComputationResult = result;
}
