#ifndef THREAD_MAPPER_H
#define THREAD_MAPPER_H

#include <windows.h>
#include <vector>
#include <string>
#include <mutex>


// Structure to hold information about a thread
struct ThreadInfo {
    DWORD threadID;                  // Thread ID
    LPVOID threadStartAddress;       
    std::vector<unsigned char> cpuFunctionData; 
};

// ThreadMapper class for mapping and transferring threads to GPU
class ThreadMapper {
public:
    // Constructor
    ThreadMapper();

    // Destructor
    ~ThreadMapper();

    // Attach to a process by name and return its handle
    HANDLE AttachToProcess(const std::string& processName);

    // Map all threads in the specified process
    bool MapThreads(const std::string& processName);

    // Retrieve thread-specific data (e.g., function data) and populate ThreadInfo
    bool RetrieveThreadData(HANDLE hProcess, ThreadInfo& threadInfo);

    // Transfer mapped thread data to the GPU
    bool TransferCPUFunctionsToGPU();

    // Accessor for GPU buffer
    unsigned char* GetGPUBuffer() const;

    // Accessor for GPU buffer size
    size_t GetGPUBufferSize() const;

    // Accessor for the number of mapped threads
    int GetMappedThreadCount() const;

private:
    std::vector<ThreadInfo> mappedThreads;  // List of mapped threads
    unsigned char* d_gpuBuffer;             // GPU buffer for storing thread data
    size_t bufferSize;                      // Size of the GPU buffer
    std::mutex threadMutex;                 // Mutex for thread-safe access to mappedThreads
};

#endif // THREAD_MAPPER_H
