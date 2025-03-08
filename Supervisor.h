#ifndef SUPERVISOR_H
#define SUPERVISOR_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "SupervisorData.h"


// Function declarations for Supervisor layer

// Initializes Supervisor data
void InitializeSupervisor(SupervisorData* supervisorData, int numThreads);

// CUDA kernel: Oversees thread management, optimizes execution
__global__ void SupervisorKernel(SupervisorData* supervisorData, int numThreads);


#endif // SUPERVISOR_H
