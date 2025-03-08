#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>       // For CUDA runtime functions
#include <device_launch_parameters.h> // For kernel parameters

#include <windows.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <set> // Include for std::set
#include <chrono>
#include <thread>
#include <TlHelp32.h>
#include <mutex> // For GPU and CPU communication
#include <iomanip> // For formatted output
#include <string>
#include <fstream>
#include <cmath> // For std::abs
#include <queue>
#include <sstream> // Include for std::ostringstream
#include "utility.h"
#include <cuda_runtime_api.h> // Updated include
#include "CodeInjector.h"
#include "GPUVerifier.h"
#include "HostVerifier.h"
#include "CUBLASManager.h"
#include <cublas_v2.h>
#include <driver_types.h>
#include "ThreadMapper.h"
#include "MatrixOps.h"
#include "MemoryBridge.h"
#include "Metadata.h"
#include "PinnedMemory.h"
#include "CUDA_SPY.h" 
#include "CudaUtils.h"
#include "SupervisorData.h"
#include "ArmageddonAlgorithm.h"

// Utility function to synchronize threads in a block
__device__ inline void synchronizeThreads() {
    __syncthreads();
}

#endif // CUDA_UTILS_H
