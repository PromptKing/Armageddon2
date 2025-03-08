#ifndef SUPERVISORDATA_H
#define SUPERVISORDATA_H

#include <vector>

// Structure to store data for each thread under supervision
struct SupervisorData {
    int threadID;                 // Unique thread ID
    float executionTime;          // Kernel execution time
    float prediction;             // Predicted result
    void* memoryRegion;           // Memory region associated with this thread
    bool isHalted;                // Whether the thread is halted

    // Additional fields for managing computations
    std::vector<float> matrixData; // Data for matrix operations
    float cpuComputationResult;    // Computation result from CPU
    float gpuComputationResult;    // Computation result from GPU
};


// Update SupervisorData after matrix multiplication
void UpdateMatrixData(SupervisorData* supervisorData, int threadID, const std::vector<float>& matrixResult);

// Update SupervisorData after CPU computation
void UpdateCPUComputation(SupervisorData* supervisorData, int threadID, float result);

// Update SupervisorData after GPU computation
void UpdateGPUComputation(SupervisorData* supervisorData, int threadID, float result);

#endif // SUPERVISORDATA_H
