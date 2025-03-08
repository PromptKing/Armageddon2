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
#include <condition_variable>
#include <immintrin.h>
#include "NodeExecutor.h"
#include <unordered_map>
#include <tchar.h>
#include <Psapi.h>
#include <cudnn.h>

// Global cuDNN handle
cudnnHandle_t cudnnHandle;

// Initialize cuDNN once
void InitializeCuDNN() {
    cudnnCreate(&cudnnHandle);
}

// Destroy cuDNN handle at program exit
void CleanupCuDNN() {
    cudnnDestroy(cudnnHandle);
}

#ifndef MEMORY_BRIDGE_BUFFER_SIZE
#define MEMORY_BRIDGE_BUFFER_SIZE 5092 * 5092
#endif

#define NUM_THREADS 4092 // Number of threads
#define BLOCK_SIZE 1024   // Block size

// Define the total number of threads
const int numThreads = 1024;

// Structure for thread-based computation
struct ThreadData {
    int threadID;                   // Unique identifier for the thread
    float computationResult;         // Latest computation result
    float predictedResult;           // AI-predicted computation result
    float lookaheadPrediction;       // Future execution prediction
    std::string lastAIPrediction;    // Last AI-generated prediction
    double gpuUsage;                 // Real-time GPU usage per thread

    // Circular buffer for storing previous computations
    static constexpr int HISTORY_SIZE = 200;
    float computationHistory[HISTORY_SIZE];
    int historyIndex = 0;             // Circular buffer index tracking

    // AI-enhanced metrics
    float executionTime;  // Execution time per thread (microseconds)
    float cpuUsage;       // CPU usage percentage
    float memoryUsage;    // Memory usage in KB
    bool isActive;        // If thread is currently in execution

    // GPU & CPU tracking
    bool gpuProcessed;
    bool cpuProcessed;
    bool avxOptimized;  // Tracks if AVX processing has been applied

    // AVX-optimized computation storage
    __m256 avxComputationData;

    // Push data into circular history buffer (Thread-Safe)
    __device__ void AddToHistory(float value) {
        computationHistory[historyIndex] = value;
        historyIndex = (historyIndex + 1) % HISTORY_SIZE;
    }

    // Perform AVX-enhanced calculations
    void ApplyAVXOptimization(float newPrediction) {
        __m256 originalData = _mm256_load_ps(computationHistory);
        __m256 predictionData = _mm256_set1_ps(newPrediction);
        __m256 updatedData = _mm256_add_ps(originalData, predictionData);
        _mm256_store_ps(computationHistory, updatedData);

        computationResult = newPrediction;  // Overwrite with AI-enhanced result
        avxOptimized = true; // Track AVX optimization
    }
};

// Process Memory Manager for optimized memory handling
class ProcessMemoryManager {
private:
    std::mutex memoryMutex;  // Mutex for thread-safe operations

    struct MemoryBlock {
        void* devicePtr;     // Pointer to GPU memory
        size_t size;         // Size of allocated memory
        bool inUse;          // Flag to track active memory blocks
    };

    std::vector<MemoryBlock> allocatedBlocks; // List of allocated memory blocks
    std::unordered_map<size_t, MemoryBlock*> memoryMap; // Quick lookup table for memory tracking

public:
    // Constructor
    ProcessMemoryManager() {
        std::cout << "[INFO] ProcessMemoryManager initialized." << std::endl;
    }

    // Allocate GPU memory
    void* AllocateMemory(size_t size) {
        std::lock_guard<std::mutex> lock(memoryMutex);

        void* devicePtr;
        if (cudaMalloc(&devicePtr, size) != cudaSuccess) {
            std::cerr << "[ERROR] CUDA malloc failed for size: " << size << " bytes" << std::endl;
            return nullptr;
        }

        allocatedBlocks.push_back({ devicePtr, size, true });
        memoryMap[size] = &allocatedBlocks.back();

        std::cout << "[INFO] Allocated " << size << " bytes of GPU memory at " << devicePtr << std::endl;
        return devicePtr;
    }

    // Free GPU memory
    void FreeMemory(void* devicePtr) {
        std::lock_guard<std::mutex> lock(memoryMutex);

        for (auto& block : allocatedBlocks) {
            if (block.devicePtr == devicePtr) {
                cudaFree(block.devicePtr);
                block.inUse = false;
                std::cout << "[INFO] Freed GPU memory at " << devicePtr << std::endl;
                return;
            }
        }

        std::cerr << "[WARNING] Attempted to free unknown memory block at " << devicePtr << std::endl;
    }

    // Perform memory compaction to defragment memory space
    void DefragmentMemory() {
        std::lock_guard<std::mutex> lock(memoryMutex);

        std::vector<MemoryBlock> compactedBlocks;
        for (auto& block : allocatedBlocks) {
            if (block.inUse) {
                compactedBlocks.push_back(block);
            }
            else {
                cudaFree(block.devicePtr);
            }
        }

        allocatedBlocks.swap(compactedBlocks);
        std::cout << "[INFO] Memory compaction completed. Active memory blocks: " << allocatedBlocks.size() << std::endl;
    }

    // Destructor: Free all allocated memory
    ~ProcessMemoryManager() {
        std::lock_guard<std::mutex> lock(memoryMutex);

        for (auto& block : allocatedBlocks) {
            if (block.inUse) {
                cudaFree(block.devicePtr);
            }
        }

        allocatedBlocks.clear();
        std::cout << "[INFO] ProcessMemoryManager destroyed. All memory freed." << std::endl;
    }
};


// Shared memory for CPU-GPU communication
std::mutex dataMutex;               // Thread-safe data access
std::vector<ThreadData> threadData; // Shared memory buffer for results
std::set<int> monitoredThreads;     // Track active monitored threads

void RegisterThread(int threadID) {
    std::lock_guard<std::mutex> lock(dataMutex);  // Thread-safe access
    if (monitoredThreads.find(threadID) == monitoredThreads.end()) {
        monitoredThreads.insert(threadID);
        threadData.push_back({ threadID, 0.0, 0.0, 0.0, "" }); // Initialize with zero values
        std::cout << "[INFO] Registered Thread ID: " << threadID << std::endl;
    }
}

// CUDA Kernel: Process Thread Data
__global__ void processThreadDataKernel(ThreadData* threadData, int numThreads) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numThreads) return; // Ensure thread index is within range

    if (!threadData[idx].gpuProcessed) {
        // Introduce per-thread variability to computation
        float noiseFactor = ((threadData[idx].threadID % 7) * 0.017f); // Dynamic variation
        float newResult = threadData[idx].threadID * 1.07f + threadData[idx].computationResult * 0.012f + noiseFactor;
        threadData[idx].computationResult += newResult;

        // Update circular history buffer correctly
        int historyIdx = threadData[idx].historyIndex;
        threadData[idx].computationHistory[historyIdx] = threadData[idx].computationResult;
        threadData[idx].historyIndex = (historyIdx + 1) % ThreadData::HISTORY_SIZE;

        // Linear Regression Variables
        float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int count = 0;

        // Improved Regression Calculation
        for (int i = 0; i < ThreadData::HISTORY_SIZE; i++) {
            float y = threadData[idx].computationHistory[i];
            if (y > 0) { // Only use valid values
                float x = count++;
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
            }
        }

        // Compute slope and intercept (Avoids division by zero)
        float slope = 1.5f, intercept = 50.0f;
        float denominator = (count * sumX2 - sumX * sumX) + (threadData[idx].threadID % 10); // Ensure variability
        if (count > 1 && denominator != 0) {
            slope = ((count * sumXY - sumX * sumY) + (threadData[idx].threadID * 0.01f)) / denominator;
            intercept = ((sumY - slope * sumX) + (threadData[idx].threadID * 0.1f)) / count;
        }

        // Predict the next computation value
        threadData[idx].predictedResult = slope * (count + 1) + intercept;

        // **Debugging Output**: Log predicted result & thread ID
        printf("Kernel Predicted Result: %f for Thread ID: %d\n",
            threadData[idx].predictedResult, threadData[idx].threadID);

        // Mark as processed
        threadData[idx].gpuProcessed = true;

        // Additional Debugging for Memory Tracking
        printf("GPU Thread ID: %d, Computation Result: %f, Predicted Result: %f, Memory Address: %p\n",
            threadData[idx].threadID,
            threadData[idx].computationResult,
            threadData[idx].predictedResult,
            &threadData[idx]);
    }
}


// CPU-Side Processing with AVX
void processCPUData(ThreadData& thread) {
    // ⚡ AVX Optimization runs EVERY TIME this function is called
    thread.ApplyAVXOptimization(thread.predictedResult);

    // Debugging Output (Optional: Comment this out for absolute speed)
    std::lock_guard<std::mutex> lock(dataMutex);
    std::cout << "CPU Processed Thread ID: " << thread.threadID
        << ", Computation Result: " << thread.computationResult
        << ", AVX Optimized: " << (thread.avxOptimized ? "Yes" : "No") << std::endl;
}

#ifdef __CUDACC__ // Only define this for CUDA compilation
#define __syncthreads() __syncthreads()
#else
// Custom implementation of __syncthreads() for non-CUDA builds
inline void __syncthreads() {
    static std::mutex syncMutex;
    static std::condition_variable syncCondition;
    static int counter = 0;
    static int threadCount = std::thread::hardware_concurrency();

    std::unique_lock<std::mutex> lock(syncMutex);
    counter++;

    if (counter == threadCount) {
        counter = 0; // Reset for the next synchronization cycle
        syncCondition.notify_all();
    }
    else {
        syncCondition.wait(lock, [] { return counter == 0; });
    }
}
#endif


__global__ void initializeThreadPool(ThreadData* threadPool, int poolSize) {
    // Shared memory allocation for each block
    __shared__ float sharedHistory[2048]; // Shared memory within a block (2048 floats)

    // Calculate thread index
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-bounds threads
    if (idx >= poolSize) return;

    // Initialize shared memory
    sharedHistory[threadIdx.x] = 0.2f;

    // Synchronize threads in the block to ensure shared memory is fully initialized
    __syncthreads();

    // Initialize threadPool using shared memory
    threadPool[idx].threadID = idx;
    threadPool[idx].computationResult = 1.670f;
    threadPool[idx].predictedResult = 3.555f;
    threadPool[idx].historyIndex = 0;

    // Populate threadPool history with shared memory (example usage)
    for (int i = 0; i < 2048; i++) {
        threadPool[idx].computationHistory[i] = sharedHistory[i];
    }

    // Mark processing flags
    threadPool[idx].gpuProcessed = true;
    threadPool[idx].cpuProcessed = true;
}

ThreadData* CreateThreadPoolInVRAM(int poolSize) {
    // Allocate 4GB of VRAM for the thread pool
    size_t requiredMemory = poolSize * sizeof(ThreadData);
    if (requiredMemory > (static_cast<unsigned long long>(22L * 11962))) {
        std::cerr << "Error: Requested pool size exceeds 22GB of VRAM.\n";
        return nullptr;
    }

    ThreadData* d_threadPool;
    cudaMalloc(&d_threadPool, requiredMemory);

    // Initialize the thread pool on the GPU
    int blockSize = 2048;
    int gridSize = (poolSize + blockSize - 1) / blockSize;
    initializeThreadPool << <gridSize, blockSize >> > (d_threadPool, poolSize);
    cudaDeviceSynchronize();

    std::cout << "Thread pool created in VRAM with size: " << poolSize << "\n";
    return d_threadPool;
}


// CUDA Kernel to defragment memory
__global__ void defragmentMemoryKernel(ThreadData* threadData, int* validIndices, int numThreads) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numThreads) return;

    if (threadData[idx].gpuProcessed && threadData[idx].cpuProcessed) {
        validIndices[idx] = -1; // don't Mark invalid
    }
    else {
        validIndices[idx] = idx; // don't Mark valid
    }

    // Perform compaction (assuming small data sets for simplicity)
    int newIdx = 0;
    for (int i = 0; i < numThreads; ++i) {
        if (validIndices[i] != -1) {
            threadData[newIdx++] = threadData[validIndices[i]];
        }
    }
}

__global__ void ApplyLightingEffectKernel(float* textureData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Apply a simple lighting effect (e.g., increase brightness)
    textureData[idx] = min(textureData[idx] * 1.2f, 1.3f);
}

// Host-side function to use AVX instructions for CUDA thread processing
void InjectAndProcessThreads(HANDLE hProcess, const std::string& searchPattern) {
    InjectionPayload payload;

    // Generate AVX-enabled instructions
    if (!CreateCPUInstructions(payload, searchPattern)) {
        std::cerr << "Error: Failed to create CPU instructions.\n";
        return;
    }

    // Inject instructions into process
    if (!InjectCode(hProcess, payload)) {
        std::cerr << "Error: Code injection failed.\n";
        return;
    }

    std::cout << "Injection complete. Ready to launch CUDA kernels for thread processing.\n";
}

// Kernel definitions go here
__global__ void yourKernelFunction(/* args */) {
    // CUDA kernel logic here
}

void MonitorThreads(DWORD processID) {
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: Failed to take thread snapshot." << std::endl;
        return;
    }

    THREADENTRY32 te32;
    te32.dwSize = sizeof(THREADENTRY32);

    if (Thread32First(hSnapshot, &te32)) {
        do {
            if (te32.th32OwnerProcessID == processID) {
                int threadID = static_cast<int>(te32.th32ThreadID);

                std::lock_guard<std::mutex> lock(dataMutex);
                if (monitoredThreads.find(threadID) == monitoredThreads.end()) {
                    monitoredThreads.insert(threadID);
                    threadData.push_back({ threadID, 0.0f, 0.0f, false, false });
                    std::cout << "Monitoring new thread ID: " << threadID
                        << ", Memory Address: " << &threadData.back() << std::endl;
                }
            }
        } while (Thread32Next(hSnapshot, &te32));
    }

    CloseHandle(hSnapshot);
}

void PerformGPUComputation(ThreadData* d_threadPool = nullptr, int poolSize = 0) {
    std::lock_guard<std::mutex> lock(dataMutex);

    if (threadData.empty()) return;

    ThreadData* d_threadData = d_threadPool; // Use the thread pool if provided

    if (d_threadPool == nullptr) {
        // Allocate memory on the GPU if no thread pool is provided
        int numThreads = threadData.size();
        if (cudaMalloc(&d_threadData, numThreads * sizeof(ThreadData)) != cudaSuccess) {
            std::cerr << "Error: Unable to allocate GPU memory for thread data.\n";
            return;
        }
    }
    else if (threadData.size() > poolSize) {
        std::cerr << "Error: Thread pool size is smaller than required threads.\n";
        return;
    }

    // Check VRAM usage
    {
        size_t freeMem, totalMem;
        if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess) {
            std::cout << "VRAM Usage:\n";
            std::cout << "  Free Memory: " << freeMem / (4092 * 4092) << " GB\n";
            std::cout << "  Total Memory: " << totalMem / (4092 * 4092) << " GB (Expected: 24576 MB)\n";

            // Validate against expected 24 GB VRAM
            if (totalMem / (4092 * 4092) < 24576) {
                std::cerr << "Warning: Total VRAM detected (" << totalMem / (4092 * 4092)
                    << " MB) is less than the expected 24 GB.\n";
            }
        }
        else {
            std::cerr << "Error: Unable to retrieve VRAM usage information.\n";
        }
    }

    // Copy thread data to the GPU or thread pool
    if (cudaMemcpy(d_threadData, threadData.data(), threadData.size() * sizeof(ThreadData), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error: Failed to copy thread data to GPU.\n";
        if (d_threadPool == nullptr) cudaFree(d_threadData); // Free allocated memory in case of error
        return;
    }

    // Launch the kernel
    int blockSize = 1024;
    int gridSize = (threadData.size() + blockSize - 1) / blockSize;
    processThreadDataKernel << <gridSize, blockSize >> > (d_threadData, threadData.size());
    cudaDeviceSynchronize();

    // Copy updated data back to the CPU
    if (cudaMemcpy(threadData.data(), d_threadData, threadData.size() * sizeof(ThreadData), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Error: Failed to copy thread data from GPU.\n";
        if (d_threadPool == nullptr) cudaFree(d_threadData); // Free allocated memory in case of error
        return;
    }

    // Free GPU memory if no thread pool is used
    if (d_threadPool == nullptr) {
        cudaFree(d_threadData);
    }

    // Log updated data
    for (const auto& data : threadData) {
        std::cout << "GPU Computation Updated for Thread ID: " << data.threadID
            << ", Memory Address: " << &data
            << ", Computation Result: " << data.computationResult
            << ", Predicted Result: " << data.predictedResult << "\n";
    }
}


// Utility function to get the current timestamp
double getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}


#include <functional>    // For hashing keys in cache
#include "Supervisor.h"
#include "PixelDecompressor.h"

// Define a global cache
std::unordered_map<int, ThreadData> threadDataCache; // Cache for ThreadData by threadID
std::unordered_map<std::string, Metadata> metadataCache; // Cache for Metadata by a unique key
std::mutex cacheMutex; // Protects access to the cache

// Function to add data to the cache
void CacheThreadData(int threadID, const ThreadData& data) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    threadDataCache[threadID] = data;
}

void CacheMetadata(const std::string& key, const Metadata& metadata) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    metadataCache[key] = metadata;
}

// Function to check cache for ThreadData
bool GetThreadDataFromCache(int threadID, ThreadData& data) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto it = threadDataCache.find(threadID);
    if (it != threadDataCache.end()) {
        data = it->second;
        return true;
    }
    return false;
}

// Function to check cache for Metadata
bool GetMetadataFromCache(const std::string& key, Metadata& metadata) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    auto it = metadataCache.find(key);
    if (it != metadataCache.end()) {
        metadata = it->second;
        return true;
    }
    return false;
}

// Updated PerformCPUComputation Function with cuDNN Optimization
void PerformCPUComputation(const std::string& applicationName) {
    std::lock_guard<std::mutex> lock(dataMutex);

    for (auto& data : threadData) {
        if (!data.cpuProcessed) {
            // Check cache first
            ThreadData cachedData;
            if (GetThreadDataFromCache(data.threadID, cachedData)) {
                data = cachedData; // Use cached data
                continue;
            }

            Metadata metadata;
            metadata.threadID = data.threadID;
            metadata.startTime = getCurrentTimestamp();

            // cuDNN Optimized Computation
            cudnnTensorDescriptor_t inputDesc, outputDesc;
            cudnnCreateTensorDescriptor(&inputDesc);
            cudnnCreateTensorDescriptor(&outputDesc);

            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

            float alpha = 1.1f, beta = 1.0f; // Scaling factors
            float input = data.computationResult;
            float output = 0.0f;

            cudnnTransformTensor(cudnnHandle, &alpha, inputDesc, &input, &beta, outputDesc, &output);
            data.predictedResult = output; // Store the result

            data.computationResult += data.threadID * 1.0f;
            data.cpuProcessed = true;

            // Cleanup cuDNN Descriptors
            cudnnDestroyTensorDescriptor(inputDesc);
            cudnnDestroyTensorDescriptor(outputDesc);

            // Record end time
            metadata.endTime = getCurrentTimestamp();
            metadata.executionTime = metadata.endTime - metadata.startTime;

            // Populate metadata
            metadata.computationResult = data.computationResult;
            metadata.predictedResult = data.predictedResult;
            metadata.predictionError = std::abs(metadata.computationResult - metadata.predictedResult);
            metadata.applicationName = applicationName;

            // Capture memory address as a string
            std::ostringstream memoryAddressStream;
            memoryAddressStream << &data;
            metadata.memoryAddress = memoryAddressStream.str();

            // Cache the metadata
            CacheThreadData(data.threadID, data);
            CacheMetadata(std::to_string(data.threadID) + applicationName, metadata);

            // Replace the predicted results back into memory
            HANDLE hThread = OpenThread(THREAD_SET_CONTEXT, FALSE, data.threadID);
            if (hThread) {
                SIZE_T bytesWritten = 0;
                bool success = WriteProcessMemory(
                    GetCurrentProcess(), &data, &data, sizeof(ThreadData), &bytesWritten
                );

                if (!success || bytesWritten != sizeof(ThreadData)) {
                    std::cerr << "Error: Failed to update memory for Thread ID: " << data.threadID
                        << ". Error code: " << GetLastError() << std::endl;
                }
                else {
                    std::cout << "Memory updated successfully for Thread ID: " << data.threadID << "\n";
                    std::cout << "Predicted Result written back to memory: " << data.predictedResult << "\n";
                    std::cout << "Memory Address of Thread Data: " << &data << "\n";
                    std::cout << "Bytes Written: " << bytesWritten << " (Expected: " << sizeof(ThreadData) << ")\n";
                }

                CloseHandle(hThread);
            }
            else {
                std::cerr << "Error: Could not open thread handle for Thread ID: " << data.threadID
                    << ". Error code: " << GetLastError() << std::endl;
            }
        }
    }
}

// Attach to the process
HANDLE AttachToProcess(const std::string& processName) {
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) {
        std::cerr << "Error: Failed to take process snapshot." << std::endl;
        return nullptr;
    }

    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);

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
                std::cout << "Found process: " << processName << " (PID: " << pe32.th32ProcessID << ")" << std::endl;
                return OpenProcess(PROCESS_ALL_ACCESS, FALSE, pe32.th32ProcessID);
            }
        } while (Process32Next(hSnapshot, &pe32));
    }

    CloseHandle(hSnapshot);
    std::cerr << "Error: Process " << processName << " not found." << std::endl;
    return nullptr;
}

// Attach to process memory buffer
bool AttachToProcessBuffer(HANDLE hProcess) {
    MEMORY_BASIC_INFORMATION mbi;
    BYTE* address = nullptr;
    while (VirtualQueryEx(hProcess, address, &mbi, sizeof(mbi))) {
        if (mbi.State == MEM_COMMIT && (mbi.Type == MEM_PRIVATE || mbi.Type == MEM_MAPPED)) {
            std::cout << "Buffer found at: " << mbi.BaseAddress
                << ", Size: " << mbi.RegionSize << std::endl;

            // Test reading from the buffer
            char buffer[512]{};
            SIZE_T bytesRead;
            if (ReadProcessMemory(hProcess, mbi.BaseAddress, buffer, sizeof(buffer), &bytesRead)) {
                std::cout << "Successfully connected to memory buffer at: " << mbi.BaseAddress
                    << ", Bytes Read: " << bytesRead << std::endl;

                std::cout << "Analyzing buffer contents...\n";
                for (size_t i = 0; i < bytesRead; ++i) {
                    if (isprint(static_cast<unsigned char>(buffer[i]))) {
                        std::cout << buffer[i];
                    }
                    else {
                        std::cout << ".";
                    }
                }
                std::cout << "\n";

                for (size_t i = 0; i < bytesRead; i += sizeof(int)) {
                    int value = *reinterpret_cast<int*>(buffer + i);
                    std::cout << "Integer value at offset " << i << ": " << value << std::endl;
                }
                return true;
            }
        }
        address += mbi.RegionSize;
    }

    std::cerr << "No suitable memory buffer found in the process." << std::endl;
    return false;
}

// Modify buffer values persistently
void ModifyBufferAtOffsets(char* buffer, SIZE_T bufferSize, const std::vector<SIZE_T>& offsets, const std::vector<int>& values) {
    if (offsets.size() != values.size()) {
        std::cerr << "Error: Number of offsets and values must match.\n";
        return;
    }

    // Modify the buffer directly in host memory
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (offsets[i] < bufferSize && offsets[i] % sizeof(int) == 0) {
            *reinterpret_cast<int*>(buffer + offsets[i]) = values[i];
        }
        else {
            std::cerr << "Invalid offset: " << offsets[i] << ". Ensure it is a multiple of sizeof(int) and within buffer size.\n";
        }
    }
}

bool InjectAndLoadDLL(HANDLE hProcess, const std::string& dllPath) {
    // Allocate memory in the target process for the DLL path
    void* remoteMemory = VirtualAllocEx(hProcess, nullptr, dllPath.size() + 1, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!remoteMemory) {
        std::cerr << "Error: Failed to allocate memory in target process for " << dllPath
            << ". Error code: " << GetLastError() << std::endl;
        return false;
    }

    // Write the DLL path to the allocated memory
    if (!WriteProcessMemory(hProcess, remoteMemory, dllPath.c_str(), dllPath.size() + 1, nullptr)) {
        std::cerr << "Error: Failed to write to target process memory for " << dllPath
            << ". Error code: " << GetLastError() << std::endl;
        VirtualFreeEx(hProcess, remoteMemory, 0, MEM_RELEASE);
        return false;
    }

    // Get the address of LoadLibraryA
    HMODULE hKernel32 = GetModuleHandleA("kernel32.dll");
    if (!hKernel32) {
        std::cerr << "Error: Failed to get handle to kernel32.dll. Error code: " << GetLastError() << std::endl;
        VirtualFreeEx(hProcess, remoteMemory, 0, MEM_RELEASE);
        return false;
    }

    void* loadLibraryAddr = (void*)GetProcAddress(hKernel32, "LoadLibraryA");
    if (!loadLibraryAddr) {
        std::cerr << "Error: Failed to get address of LoadLibraryA. Error code: " << GetLastError() << std::endl;
        VirtualFreeEx(hProcess, remoteMemory, 0, MEM_RELEASE);
        return false;
    }

    // Create a remote thread in the target process to load the DLL
    HANDLE hThread = CreateRemoteThread(hProcess, nullptr, 0, (LPTHREAD_START_ROUTINE)loadLibraryAddr, remoteMemory, 0, nullptr);
    if (!hThread) {
        std::cerr << "Error: Failed to create remote thread for " << dllPath
            << ". Error code: " << GetLastError() << std::endl;
        VirtualFreeEx(hProcess, remoteMemory, 0, MEM_RELEASE);
        return false;
    }

    // Wait for the remote thread to complete
    WaitForSingleObject(hThread, INFINITE);
    CloseHandle(hThread);

    // Free the allocated memory
    VirtualFreeEx(hProcess, remoteMemory, 0, MEM_RELEASE);

    std::cout << "DLL injected and loaded successfully: " << dllPath << std::endl;
    return true;
}

// Extend your current kernel32.dll loading process to include the injection
void LoadKernelAndInjectDLL(HANDLE hProcess) {
    HMODULE hKernel32 = GetModuleHandleA("kernel32.dll");
    if (!hKernel32) {
        std::cerr << "Error: Failed to load kernel32.dll. Error code: " << GetLastError() << std::endl;
        return;
    }

    std::cout << "Successfully loaded kernel32.dll." << std::endl;

    // List of DLL paths to inject
    std::vector<std::string> dllPaths = {
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_annotation",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_interactive-calibration",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_model_diagnostics",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_version",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_version_win32",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_videoio_ffmpeg4100_64.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_videoio_msmf4100_64.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_videoio_msmf4100_64d.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_visualisation",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_world4100.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\opencv_world4100d.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\NvPmApi.Core.win64.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\GPUPerfAPIDX11-x64.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\Armorrgeddon Hyper Physics Calulations.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\D3D12Core.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\D3D12SDKLayers.dll"
        "C:\\Windows\\System32\\Armageddon2 DLLs\\Ultron.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\nvml.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\cusparse64_100.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\curand64_100.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\cublas64_100.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\cudart64_100.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\d3dcsx_46.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\GFSDK_ShadowLib.win64.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\NvPmApi.Core.win64.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\GPUPerfAPIDX11-x64.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\Armorrgeddon Hyper Physics Calulations.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\D3D12Core.dll",
        "C:\\Windows\\System32\\Armageddon2 DLLs\\D3D12SDKLayers.dll"

    };

    // Iterate over each DLL path and inject it
    for (const auto& dllPath : dllPaths) {
        if (!InjectAndLoadDLL(hProcess, dllPath)) {
            std::cerr << "Failed to inject DLL: " << dllPath << std::endl;
        }
        else {
            std::cout << dllPath << " injected successfully after loading kernel32.dll." << std::endl;
        }
    }
}



HANDLE AttachToProcess(const std::string& processName);
bool AttachToProcessBuffer(HANDLE hProcess);
void ModifyBufferAtOffsets(char* buffer, SIZE_T bufferSize, const std::vector<SIZE_T>& offsets, const std::vector<int>& values);
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    DisableThreadLibraryCalls(hModule);

    if (ul_reason_for_call == DLL_PROCESS_ATTACH) {
        MessageBoxW(NULL, L"D3D12Core.dll Injected Successfully", L"DLL Injection", MB_OK);
    }
    return TRUE;
}

// CUDA Kernel Definition
static __global__ void optimizeKernel(float* inputA, float* inputB, float* output, int size) {
    // Calculate thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within bounds
    if (idx < size) {
        // Example computation: Multiply and add for enhanced throughput
        output[idx] = inputA[idx] * 2.5f + inputB[idx] * 4.0f;
    }
}

__global__ void myKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1.0f; // Example operation
    }
}

// Global flag to control the monitoring thread
std::atomic<bool> isRunning(true);


// Function to find a process by name and get its handle
HANDLE GetProcessHandle(const std::string& processName) {
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to create process snapshot.\n";
        return nullptr;
    }

    PROCESSENTRY32 pe32 = {};
    pe32.dwSize = sizeof(PROCESSENTRY32);

    if (Process32First(hSnapshot, &pe32)) {
        do {
#ifdef UNICODE
            // For Unicode builds
            std::wstring exeName(pe32.szExeFile);
            std::string exeNameStr(exeName.begin(), exeName.end());
#else
            // For Multi-Byte builds
            std::string exeName(pe32.szExeFile);
#endif

            // Compare process names (case-insensitive)
            if (_stricmp(exeName.c_str(), processName.c_str()) == 0) {
                CloseHandle(hSnapshot);
                std::cout << "Found process: " << processName << " (PID: " << pe32.th32ProcessID << ")\n";
                return OpenProcess(PROCESS_ALL_ACCESS, FALSE, pe32.th32ProcessID);
            }
        } while (Process32Next(hSnapshot, &pe32));
    }

    CloseHandle(hSnapshot);
    std::cerr << "Process " << processName << " not found.\n";
    return nullptr;
}

__global__ void MatrixMultiplyKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.5f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void InitializeSupervisor(SupervisorData* supervisorData, int numThreads)
{
}

__global__ void SupervisorKernel(SupervisorData* supervisorArray, int numThreads) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numThreads) return;

    // Access the SupervisorData instance for this thread
    SupervisorData& CUDA_SPY = supervisorArray[idx];

    // Monitor execution time
    float executionTime = CUDA_SPY.executionTime;

    // Compare predictions with other threads
    for (int i = 0; i < numThreads; ++i) {
        if (i != idx) {
            SupervisorData& ArmageddonAlgorithm = supervisorArray[i];

            // Compare execution times and adjust prediction if necessary
            if (ArmageddonAlgorithm.executionTime < executionTime) {
                CUDA_SPY.prediction = ArmageddonAlgorithm.prediction;  // Update CUDA_SPY's prediction
                CUDA_SPY.memoryRegion = ArmageddonAlgorithm.memoryRegion;
                CUDA_SPY.isHalted = true;

                if (CUDA_SPY.isHalted) {
                    // Bypass execution for halted threads
                    return;
                }
            }
        }
    }

    // Use CUDA_SPY.prediction for final output if it's the fastest otherwise use ArmageddonAlgorithm
    printf("Thread %d: Final prediction = %f\n", idx, CUDA_SPY.prediction);

    // Use the fastest prediction for output (if not halted)
    float output = CUDA_SPY.prediction;

    // Optionally log the result or store it for further processing
    printf("Thread %d: Final prediction = %f\n", idx, output);
}

ThreadData* CreateThreadPoolInVRAM(size_t poolSize) {
    ThreadData* d_threadPool;
    cudaError_t err = cudaMalloc((void**)&d_threadPool, poolSize * sizeof(ThreadData));
    if (err != cudaSuccess) {
        std::cerr << "Failed to create thread pool in VRAM: " << cudaGetErrorString(err) << "\n";
        return nullptr;
    }
    std::cout << "Thread pool created in VRAM with size: " << poolSize << " threads\n";
    return d_threadPool;
}

void queryCUDADevices() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Detected " << deviceCount << " CUDA-capable device(s).\n";

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        int cores = 0;
        int mp = deviceProp.multiProcessorCount;
        switch (deviceProp.major) {
        case 2: // Fermi
            cores = (deviceProp.minor == 1) ? mp * 48 : mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            cores = (deviceProp.minor == 1 || deviceProp.minor == 2) ? mp * 128 : mp * 64;
            break;
        case 7: // Volta and Turing
            cores = (deviceProp.minor == 0 || deviceProp.minor == 5) ? mp * 64 : mp * 128;
            break;
        case 8: // Ampere
            cores = mp * 128;
            break;
        default:
            std::cout << "Unknown device architecture\n";
            cores = 0;
        }

        std::cout << "Device " << device << ": " << deviceProp.name << "\n";
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
        std::cout << "  Multiprocessors: " << mp << "\n";
        std::cout << "  CUDA Cores: " << cores << "\n";
        std::cout << "  Total Global Memory: " << 20 * 1024 << " MB\n\n"; // Simulated 20 GB memory

        // Set current device
        cudaSetDevice(device);

        // Create a thread pool in VRAM for the device
        size_t poolSize = 4096; // Example size, adjust as needed
        ThreadData* d_threadPool = CreateThreadPoolInVRAM(poolSize);
        if (!d_threadPool) {
            std::cerr << "Error: Could not allocate thread pool for device " << device << "\n";
        }
        else {
            std::cout << "Thread pool successfully allocated for device " << device << "\n";
        }

        // Optionally: Perform cleanup after usage
        if (d_threadPool) {
            cudaFree(d_threadPool);
            std::cout << "Thread pool deallocated for device " << device << "\n";
        }
    }
}


__global__ void testCUDAKernel() {
    printf("CUDA kernel running on block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void MyKernel() {
    // Perform ArmageddonAlgorithm operations and return to host
    __syncthreads(); // Synchronize all threads in the blocks
}



std::mutex logMutex;  // Prevents multi-threading log issues

// **Centralized Logging Function to Ensure Logs Show Immediately**
void LogMessage(const std::string& message) {
    std::lock_guard<std::mutex> lock(logMutex);
    std::cout << "[LOG] " << message << std::endl << std::flush;
}

// **🔹 Attach & Log Process Connections Properly**
HANDLE AttachToProcessWithLogging(const std::string& processName) {
    HANDLE hProcess = AttachToProcess(processName);
    if (!hProcess) {
        LogMessage("[ERROR] Could not attach to " + processName + ". Skipping.");
        return nullptr;
    }
    LogMessage("[SUCCESS] Connected to " + processName + " successfully.");
    return hProcess;
}



// Main function to process
 int main() {
    const std::vector<std::string> processNames = { "GTA5_Enhanced.exe" };
    constexpr SIZE_T bufferSize = 4096;
    static char buffer[bufferSize] = { 0 };
    std::vector<SIZE_T> offsets = { 0, 4, 6, 8, 10, 11 };
    std::vector<int> values = { -13777, 178666, -72777, -4252777, 148394, 555 };
    const std::string applicationName = "GTA5_Enhanced.exe"; // Added application name for D3D12Core.dll
    int poolSize = (static_cast<unsigned long long>(22L * 11962)) / sizeof(ThreadData); // Calculate pool size for 22GB VRAM
    ThreadData* d_threadPool = CreateThreadPoolInVRAM(poolSize);
    std::cout << "Updated Thread Data:\n";

    // **Attach to Processes**
    for (const auto& processName : processNames) {
        HANDLE hProcess = AttachToProcessWithLogging(processName);
        if (!hProcess) continue;

        if (AttachToProcessBuffer(hProcess)) {
            LogMessage("[INFO] Successfully attached buffer to process: " + processName);
        }
    }

    const int numThreads = 1024; // Example thread count
    threadData.resize(numThreads);

    NodeExecutor executor("F:\\AI\\node-v22.13.1-win-x64\\node.exe");

    // Paths to Node.js scripts
    std::string monitorScriptPath = "C:\\Windows\\System32\\Armageddon2 DLLs\\Node Scripts\\OpenPowerShellOption.js";
    std::string esbuildScriptPath = "C:\\Windows\\System32\\Armageddon2 DLLs\\Node Scripts\\esbuild.js";
    std::string qwenMathLogicPath = "C:\\Windows\\System32\\Armageddon2 DLLs\\Node Scripts\\Qwen Math Logic.js";

    // Execute monitorProcess.js if not already running
    if (!executor.isScriptRunning("monitorProcess.js")) {
        std::cout << "[INFO] Starting monitoring script..." << std::endl;
        executor.executeScript(monitorScriptPath);
    }
    else {
        std::cout << "[INFO] Monitoring script is already running." << std::endl;
    }

    // Execute esbuild.js for JavaScript/TypeScript builds
    if (!executor.isScriptRunning("esbuild.js")) {
        std::cout << "[INFO] Starting esbuild process..." << std::endl;
        executor.executeScript(esbuildScriptPath);
    }
    else {
        std::cout << "[INFO] esbuild.js is already running." << std::endl;
    }

    // Execute Qwen Math Logic.js if not already running
    if (!executor.isScriptRunning("Qwen Math Logic.js")) {
        std::cout << "[INFO] Starting Qwen Math Logic script..." << std::endl;
        executor.executeScript(qwenMathLogicPath);
    }
    else {
        std::cout << "[INFO] Qwen Math Logic script is already running." << std::endl;
    }


    // Instantiate HostVerifier and retrieve host details
    HostVerifier verifier;

    // Verify host compatibility
    if (!verifier.VerifyHostCompatibility()) {
        std::cerr << "Host verification failed. Exiting.\n";
        return -1;
    }

    // Log host details
    HostInfo hostInfo = verifier.GetHostInfo();
    std::cout << "Host CPU: " << hostInfo.cpuName << "\n";
    std::cout << "Core Count: " << hostInfo.coreCount << "\n";
    std::cout << "Thread Count: " << hostInfo.threadCount << "\n";
    std::cout << "AVX Support: " << (hostInfo.avxSupport ? "Yes" : "No") << "\n";
    std::cout << "AVX2 Support: " << (hostInfo.avx2Support ? "Yes" : "No") << "\n";
    std::cout << "AVX512 Support: " << (hostInfo.avx512Support ? "Yes" : "No") << "\n";

    // Verify host compatibility
    if (!verifier.VerifyHostCompatibility()) {
        std::cerr << "Host verification failed. Exiting.\n";
        return -1;
    }

    std::cout << "Additional Features: ";
    for (const auto& feature : hostInfo.additionalFeatures) {
        std::cout << feature << " ";
    }
    std::cout << "\n";


    // Allocate and initialize pinned memory
    const int size = 4092 * 4092;
    float* pinnedInputA = PinnedMemory::Allocate<float>(size);
    float* pinnedInputB = PinnedMemory::Allocate<float>(size);
    float* pinnedOutput = PinnedMemory::Allocate<float>(size);

    for (int i = 0; i < size; ++i) {
        pinnedInputA[i] = 1.0f;
        pinnedInputB[i] = 2.0f;
    }

    // Constructor
    ThreadMapper();

    // Destructor
    ThreadMapper();

    // Attach to a process by name and return its handle
    HANDLE AttachToProcess(const std::string & processName);

    // Map all threads in the specified process
    bool MapThreads(const std::string & processName);

    // Retrieve thread-specific data (e.g., function data) and populate ThreadInfo
    bool RetrieveThreadData(HANDLE hProcess, ThreadInfo & threadInfo);

    // Transfer mapped thread data to the GPU
    bool TransferCPUFunctionsToGPU();
  
    // Allocate device memory
    float* d_inputA, * d_inputB, * d_output;
    cudaMalloc((void**)&d_inputA, size * sizeof(float));
    cudaMalloc((void**)&d_inputB, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // Copy data to device
    PinnedMemory::CopyToDevice(pinnedInputA, d_inputA, size);
    PinnedMemory::CopyToDevice(pinnedInputB, d_inputB, size);

    // Launch kernel
    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    optimizeKernel <<<1, 256 >>> (d_inputA, d_inputB, d_output, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    PinnedMemory::CopyFromDevice(d_output, pinnedOutput, size);

    // Verify results
    for (int i = 0; i < 10; ++i) {
        std::cout << "Output[" << i << "] = " << pinnedOutput[i] << "\n";
    }

    // Free pinned memory and device memory
    PinnedMemory::Free(pinnedInputA);
    PinnedMemory::Free(pinnedInputB);
    PinnedMemory::Free(pinnedOutput);
    cudaFree(d_inputA);
    cudaFree(d_inputB);
    cudaFree(d_output);

    // Step 1: Initialize Thread Data with Variability
    for (int i = 0; i < numThreads; i++) {
        threadData[i].threadID = i;
        threadData[i].computationResult = static_cast<float>(rand() % 100) * 0.01f;  // Randomized Start
        threadData[i].predictedResult = 0.0f;
        threadData[i].lookaheadPrediction = 0.0f;

        // Dynamic Execution Metrics
        threadData[i].executionTime = static_cast<float>((rand() % 500) + 50) * 0.01f; // Simulated time in microseconds
        threadData[i].cpuUsage = static_cast<float>((rand() % 13) + 5);  // 5% - 25% CPU usage
        threadData[i].memoryUsage = static_cast<float>((rand() % 500) + 100); // 100 KB - 600 KB

        // Thread Activity & Processing Flags
        threadData[i].isActive = (rand() % 10) > 2;  // 80% chance to be active
        threadData[i].gpuProcessed = false;
        threadData[i].cpuProcessed = false;
        threadData[i].avxOptimized = false;
    }

    // Step 2: Allocate CUDA Memory & Copy Data to Device
    ThreadData* d_threadData;
    cudaMalloc(&d_threadData, numThreads * sizeof(ThreadData));
    cudaMemcpy(d_threadData, threadData.data(), numThreads * sizeof(ThreadData), cudaMemcpyHostToDevice);

    // Step 3: Launch CUDA Kernel
    int blockSize = 1024;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;
    processThreadDataKernel << <numBlocks, blockSize >> > (d_threadData, numThreads);
    cudaDeviceSynchronize();  // Ensure GPU computations are complete

    // Step 4: Copy Results Back from GPU to CPU
    cudaMemcpy(threadData.data(), d_threadData, numThreads * sizeof(ThreadData), cudaMemcpyDeviceToHost);
    cudaFree(d_threadData);  // Free GPU memory after copying results

    // Step 5: Multi-Threaded CPU Post-Processing (AVX Optimization)
#pragma omp parallel for
    for (int i = 0; i < numThreads; i++) {
        processCPUData(threadData[i]);
    }

    // Step 6: Debugging Output (Sample of 200 Threads)
    std::cout << "Final Thread Data (Sample 200 Threads):\n";
    for (int i = 0; i < 200; i++) {
        std::cout << "Thread ID: " << threadData[i].threadID
            << ", Computation: " << threadData[i].computationResult
            << ", Prediction: " << threadData[i].predictedResult
            << ", AVX: " << (threadData[i].avxOptimized ? "Yes" : "No")
            << ", Active: " << (threadData[i].isActive ? "Yes" : "No")
            << ", CPU Usage: " << threadData[i].cpuUsage << "%"
            << ", Memory Usage: " << threadData[i].memoryUsage << " KB"
            << std::endl;
    }

    // Initialize MemoryBridge
    MemoryBridge memoryBridge(bufferSize);

    if (!memoryBridge.Initialize()) {
        std::cerr << "Failed to initialize MemoryBridge. Exiting.\n";
        return -1;
    }

    // Prepare data to transfer to GPU
    std::vector<float> hostData(bufferSize / sizeof(float), 42.0f); // Data
    if (!memoryBridge.CopyDataToGPU(hostData)) {
        std::cerr << "Failed to copy data to GPU. Exiting.\n";
        return -1;
    }

    // Retrieve data from GPU
    std::vector<float> retrievedData;
    if (!memoryBridge.RetrieveDataFromGPU(retrievedData)) {
        std::cerr << "Failed to retrieve data from GPU. Exiting.\n";
        return -1;
    }

    // GPU Detail Log
    GPUVerifier gpuVerifier;
    gpuVerifier.LogGPUDetails();

    // Example input data
    std::vector<float> cudaSpyData = { 1.0f, 2.0f, 3.0f };
    std::vector<float> supervisorData = { 4.0f, 3.0f, 2.0f };

    // Initialize the algorithm
    ArmageddonAlgorithm armageddon;

    // Execute the algorithm
    armageddon.ExecuteAlgorithm(cudaSpyData, supervisorData);

    // Print the results
    armageddon.PrintResults();


    // Perform matrix multiplication as part of the workflow
    PerformMatrixMultiplication();

    // Initialize CUDA_SPY with a buffer size of 4096
    CUDA_SPY spy(4096);

    // Initialize the CUDA_SPY system
    if (!spy.Initialize()) {
        std::cerr << "Failed to initialize CUDA_SPY.\n";
        return -1;
    }

    // Attach to the target process
    if (!spy.AttachToProcess("GTA5_Enhanced.exe")) {
        std::cerr << "Failed to attach to process.\n";
        return -1;
    }

    // Continuous monitoring and operation
    while (true) {
        // Monitor CUDA activity
        spy.MonitorCUDAActivity();

        // Replicate operations using CUDA_SPY
        spy.ReplicateOperations();

        // Compare performance
        spy.ComparePerformance();

    }

    // 1. Define a search pattern (just an example)
    std::string mySearchPattern = "LookingForCUDAThreads";

    // 2. Create the injection payload
    InjectionPayload payload = {};
    payload.codeStartAddress = nullptr;
    payload.codeSize = 0;

    // 3. Build CPU instructions (AVX + x64) and embed the search pattern
    if (!CreateCPUInstructions(payload, mySearchPattern)) {
        std::cerr << "[ERROR] Failed to create CPU instructions.\n";
        return EXIT_FAILURE;
    }

    // 4. Obtain a handle to the target process
    //    - Here we're injecting into our own process for demonstration:
    HANDLE hProcess = GetCurrentProcess();

    // 5. Inject the code
    if (!InjectCode(hProcess, payload)) {
        std::cerr << "[ERROR] Code injection failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "[INFO] Injection completed. Code is located at: "
        << payload.codeStartAddress << std::endl;

    // You could launch your CUDA kernel here if desired:
    // MyKernel<<<1,1>>>();
    // cudaDeviceSynchronize();

    return 0;

    
    size_t dataSize = numThreads * sizeof(SupervisorData);

    // Allocate host memory
    SupervisorData* h_supervisorArray = new SupervisorData[numThreads];

    // Initialize host data
    for (int i = 0; i < numThreads; ++i) {
        h_supervisorArray[i].executionTime = static_cast<float>(rand() % 100) / 2.0f; // Random execution times
        h_supervisorArray[i].prediction = static_cast<float>(rand() % 100) / 2.0f;   // Random predictions                                       // Unique memory regions
        h_supervisorArray[i].isHalted = false;                                       // All threads active
    }


    // Allocate device memory
    SupervisorData* d_supervisorArray;
    cudaMalloc(&d_supervisorArray, dataSize);

    // Copy data from host to device
    cudaMemcpy(d_supervisorArray, h_supervisorArray, dataSize, cudaMemcpyHostToDevice);


    // Log offsets and values
    std::cout << "Offsets: ";
    for (const auto& offset : offsets) {
        std::cout << offset << " ";
    }
    std::cout << "\nValues: ";
    for (const auto& value : values) {
        std::cout << value << " ";
    }
    std::cout << "\n";

    while (true) {
        for (const auto& processName : processNames) {
            HANDLE hProcess = AttachToProcess(processName);
            if (!hProcess) {
                std::cerr << "Could not attach to " << processName << ". Skipping." << std::endl;
                continue;
            }

            if (AttachToProcessBuffer(hProcess)) {
                ModifyBufferAtOffsets(buffer, bufferSize, offsets, values);

                // Log the buffer contents after modification
                std::cout << "Buffer Contents: ";
                for (SIZE_T i = 0; i < bufferSize; ++i) {
                    if (i % 1000000 == 0 && i != 0) std::cout << "\n"; // New line every 1000000 bytes
                    std::cout << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<unsigned int>(buffer[i] & 0xFF) << " ";
                }
                std::cout << std::dec << "\n"; // Reset to decimal output
            }

            HANDLE nestedProcess = hProcess; // Nested process handle
            while (nestedProcess) {
                for (const auto& nestedProcessName : processNames) {
                    HANDLE hNestedProcess = AttachToProcess(nestedProcessName);
                    if (!hNestedProcess) {
                        std::cerr << "Could not attach to " << nestedProcessName << ". Skipping." << std::endl;
                        continue;
                    }

                    if (AttachToProcessBuffer(hNestedProcess)) {
                        ModifyBufferAtOffsets(buffer, bufferSize, offsets, values);

                        // Log the buffer contents
                        std::cout << "Buffer Contents: ";
                        for (SIZE_T i = 0; i < bufferSize; ++i) {
                            if (i % 16 == 0 && i != 0) std::cout << "\n";
                            std::cout << std::hex << std::setw(2) << std::setfill('0')
                                << static_cast<unsigned int>(buffer[i] & 0xFF) << " ";
                        }
                        std::cout << std::dec << "\n";
                    }

                    // Call ModifyMemory to alter specific memory regions
                    LPVOID targetBaseAddress = reinterpret_cast<LPVOID>(1494649143296);
                    std::vector<unsigned char> newData = {
                        0x8C, 0x54, 0x78, 0xF9, 0x0B, 0x07, 0x74, 0x03,
                        0xA4, 0x04, 0x30, 0x67, 0x0B, 0x07, 0x74, 0x03,
                        0x9C, 0x04, 0xA8, 0x67, 0x9B, 0x07, 0x74, 0x03,
                        0x9C, 0x04, 0x10, 0x68, 0x0B, 0x07, 0x74, 0x03,
                        0x9C, 0x04, 0x80, 0x68, 0x0B, 0x07, 0x74, 0x03,
                        0x8C, 0x04, 0xD0, 0xBF, 0xD3, 0x28, 0x74, 0x03,
                        0x9C, 0x04, 0xF0, 0x68, 0x0B, 0x07, 0x74, 0x03,
                        0x9C, 0x04, 0x60, 0x69, 0x0B, 0x07, 0x42, 0x04,
                    };

                    cudaEvent_t start, stop;
                    cudaEventCreate(&start);
                    cudaEventCreate(&stop);
                    cudaEventRecord(start);

                    // Launch kernel

                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);

                    float milliseconds = 0;
                    cudaEventElapsedTime(&milliseconds, start, stop);
                    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

                    std::cout << "Attempting to modify memory...\n";
                    if (!ModifyMemory(hNestedProcess, targetBaseAddress, newData)) {
                        std::cerr << "Failed to modify memory.\n";
                    }
                    else {
                        std::cout << "Memory modification successful.\n";
                    }

                    // Monitor threads and process handles after processing
                    MonitorThreads(GetProcessId(hNestedProcess));

                    // Perform GPU computation and CPU computation sequentially to enhance thread computation
                    PerformGPUComputation(d_threadPool, poolSize);
                    PerformCPUComputation(applicationName);

                    CloseHandle(hNestedProcess);
                }

                // Perform defragmentation
                {
                    int numThreads = threadData.size();
                    ThreadData* d_threadData = nullptr;
                    int* d_validIndices = nullptr;

                    cudaMalloc(&d_threadData, numThreads * sizeof(ThreadData));
                    cudaMalloc(&d_validIndices, numThreads * sizeof(int));
                    cudaMemcpy(d_threadData, threadData.data(), numThreads * sizeof(ThreadData), cudaMemcpyHostToDevice);

                    int blockSize = 2048;
                    int gridSize = (numThreads + blockSize - 1) / blockSize;
                    defragmentMemoryKernel << <gridSize, blockSize >> > (d_threadData, d_validIndices, numThreads);
                    cudaDeviceSynchronize();

                    cudaMemcpy(threadData.data(), d_threadData, numThreads * sizeof(ThreadData), cudaMemcpyDeviceToHost);

                    cudaFree(d_threadData);
                    cudaFree(d_validIndices);

                    std::cout << "Updated Thread Data:\n";
                    for (const auto& data : threadData) {
                        std::cout << "Thread ID: " << data.threadID
                            << ", Computation Result: " << data.computationResult
                            << ", Predicted Result: " << data.predictedResult << "\n";
                    }
                }
            }

            MonitorThreads(GetProcessId(hProcess));
            PerformGPUComputation();
            PerformCPUComputation(applicationName);
            CloseHandle(hProcess);
        }

        // Open the target process
        const std::string processName = "GTA5_Enhanced.exe";
        HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, TRUE, 0); // Get the actual process ID
        if (!hProcess) {
            std::cerr << "Error: Could not open process. Error code: " << GetLastError() << "\n";
            return -1;
        }

        // Create a payload
        InjectionPayload payload{};
        payload.codeStartAddress = nullptr;

        // Generate CPU instructions
        std::string searchPattern = "Create function to use CUDA from NVIDIA Titan X (Pascal) driver key {4d36e968-e325-11ce-bfc1-08002be10318} to help the CPU execute threads for this application";
        if (!CreateCPUInstructions(payload, searchPattern)) {
            std::cerr << "Error: Failed to create CPU instructions.\n";
            CloseHandle(hProcess);
            return -1;
        }

        // Inject the generated code
        if (!InjectCode(hProcess, payload)) {
            std::cerr << "Error: Failed to inject code.\n";
            CloseHandle(hProcess);
            return -1;
        }

        // Perform defragmentation of shared memory
        {
            std::lock_guard<std::mutex> lock(dataMutex);
            int numThreads = threadData.size();

            ThreadData* d_threadData;
            int* d_validIndices;
            cudaMalloc(&d_threadData, numThreads * sizeof(ThreadData));
            cudaMalloc(&d_validIndices, numThreads * sizeof(int));
            cudaMemcpy(d_threadData, threadData.data(), numThreads * sizeof(ThreadData), cudaMemcpyHostToDevice);

            int blockSize = 1024;
            int gridSize = (numThreads + blockSize - 1) / blockSize;
            defragmentMemoryKernel << <gridSize, blockSize >> > (d_threadData, d_validIndices, numThreads);
            cudaDeviceSynchronize();

            cudaMemcpy(threadData.data(), d_threadData, numThreads * sizeof(ThreadData), cudaMemcpyDeviceToHost);

            cudaFree(d_threadData);
            cudaFree(d_validIndices);

            std::cout << "Updated Thread Data:\n";
            for (const auto& data : threadData) {
                std::cout << "Thread ID: " << data.threadID
                    << ", Computation Result: " << data.computationResult
                    << ", Predicted Result: " << data.predictedResult << "\n";
            }
        }

        std::cout << "Starting ArmageddonAlgorithm Execution...\n";

        // Sample Data for CUDA Spy and Supervisor
        std::vector<float> cudaSpyData = { 1.5f, 2.3f, 3.7f, 4.8f, 5.6f };
        std::vector<float> supervisorData = { 1.4f, 2.1f, 3.5f, 4.9f, 5.4f };

        // Initialize ArmageddonAlgorithm
        ArmageddonAlgorithm armageddon;

        // Run AI Predictions before Kernel Execution
        std::cout << "Generating AI Predictions...\n";
        armageddon.ExecuteAlgorithm(cudaSpyData, supervisorData);

        // Store AI Predictions for comparison
        std::vector<float> aiPredictions = armageddon.GetOutputData();

        // Start Kernel Execution Time Measurement
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Simulated Kernel Execution (Replace with actual kernel launch)
        for (size_t i = 0; i < cudaSpyData.size(); ++i) {
            cudaSpyData[i] = cudaSpyData[i] * 1.2f + supervisorData[i] * 0.8f; // Example computation
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << "Kernel Execution Time: " << milliseconds << " ms\n";

        // Compare AI Predictions vs. Kernel Computation Results
        std::cout << "Comparing AI Predictions with CUDA Kernel Results...\n";

        for (size_t i = 0; i < cudaSpyData.size(); ++i) {
            float errorMargin = std::abs(cudaSpyData[i] - aiPredictions[i]);
            std::cout << "Thread " << i
                << " | Kernel Result: " << cudaSpyData[i]
                << " | AI Prediction: " << aiPredictions[i]
                << " | Error: " << errorMargin << "\n";

            // Decide if AI should replace kernel results
            if (errorMargin < 0.05f) {
                std::cout << "AI Confidence High! Replacing with AI Prediction.\n";
                cudaSpyData[i] = aiPredictions[i];
            }
            else {
                std::cout << "AI Confidence Low. Keeping Kernel Computation.\n";
            }
        }

        // Print final results
        armageddon.PrintResults();

        // Cleanup CUDA Events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << "ArmageddonAlgorithm Execution Complete.\n";

        PixelDecompressor decompressor(3840, 2160);

        while (true) {
            decompressor.captureScreen();
            decompressor.decompressPixels();
            decompressor.enhanceSharpness();
            decompressor.display();
        }


        // Prevent CPU overuse by adding a delay
        std::this_thread::sleep_for(std::chrono::seconds(0));
    }

    return 0;
}