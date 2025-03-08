#ifndef GPU_DATA_CAPTURE_H
#define GPU_DATA_CAPTURE_H

#include <string>
#include <fstream>
#include <mutex>
#include <filesystem>
#include <vector>

class GPUDataCapture {
public:
    // Constructor to initialize the cache directory and size limit
    GPUDataCapture(const std::string& cacheDir = "F:\\Armageddon Cache", size_t maxCacheSize = 100 * 1024 * 1024 * 1024);

    // Destructor to clean up resources
    ~GPUDataCapture();

    // Initialize the GPU data capture system
    bool Initialize();

    // Log data to the current log file
    bool LogData(const std::string& data);

    // Enforce cache size limit
    void EnforceCacheLimit();

private:
    // Directory for storing cache files
    std::string cacheDirectory;

    // Maximum cache size in bytes
    size_t maxCacheSize;

    // Current log file stream
    std::ofstream currentLogFile;

    // Path of the currently active log file
    std::filesystem::path currentLogFilePath;

    // Mutex for thread-safe logging
    std::mutex logMutex;

    // Generate a new log file name based on the current timestamp
    std::filesystem::path GenerateLogFileName();

    // Get the total size of the cache directory
    size_t GetCurrentCacheSize();
};

#endif // GPU_DATA_CAPTURE_H
