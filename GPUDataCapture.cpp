#include "GPUDataCapture.h"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

// Constructor
GPUDataCapture::GPUDataCapture(const std::string& cacheDir, size_t maxCacheSize)
    : cacheDirectory(cacheDir), maxCacheSize(maxCacheSize) {
}

// Destructor
GPUDataCapture::~GPUDataCapture() {
    if (currentLogFile.is_open()) {
        currentLogFile.close();
    }
}

// Initialize the data capture system
bool GPUDataCapture::Initialize() {
    try {
        // Create the cache directory if it does not exist
        if (!fs::exists(cacheDirectory)) {
            fs::create_directories(cacheDirectory);
        }

        // Open the first log file
        currentLogFilePath = GenerateLogFileName();
        currentLogFile.open(currentLogFilePath, std::ios::out | std::ios::app);
        if (!currentLogFile.is_open()) {
            std::cerr << "Error: Failed to open log file in " << cacheDirectory << std::endl;
            return false;
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing GPUDataCapture: " << e.what() << std::endl;
        return false;
    }
}

// Log data to the current log file
bool GPUDataCapture::LogData(const std::string& data) {
    std::lock_guard<std::mutex> lock(logMutex);

    try {
        // Rotate the log file if its size exceeds 100 MB
        if (currentLogFile.tellp() >= (100 * 1024 * 1024)) {
            currentLogFile.close();
            currentLogFilePath = GenerateLogFileName();
            currentLogFile.open(currentLogFilePath, std::ios::out | std::ios::app);
            if (!currentLogFile.is_open()) {
                std::cerr << "Error: Failed to open new log file." << std::endl;
                return false;
            }
        }

        // Write data to the log file
        currentLogFile << data << std::endl;
        currentLogFile.flush(); // Ensure the data is written immediately

        // Enforce the cache size limit
        EnforceCacheLimit();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error logging data: " << e.what() << std::endl;
        return false;
    }
}

// Enforce the cache size limit by removing the oldest files
void GPUDataCapture::EnforceCacheLimit() {
    size_t currentSize = GetCurrentCacheSize();
    if (currentSize <= maxCacheSize) return;

    try {
        // Sort files by modification time to delete the oldest ones first
        std::vector<fs::directory_entry> files;
        for (const auto& entry : fs::directory_iterator(cacheDirectory)) {
            if (fs::is_regular_file(entry)) {
                files.push_back(entry);
            }
        }

        // Sort files by last write time
        std::sort(files.begin(), files.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
            return fs::last_write_time(a) < fs::last_write_time(b);
            });

        // Remove files until the cache size is within the limit
        for (const auto& file : files) {
            if (file.path() == currentLogFilePath) continue; // Compare paths
            fs::remove(file);
            currentSize = GetCurrentCacheSize();
            if (currentSize <= maxCacheSize) break;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error enforcing cache limit: " << e.what() << std::endl;
    }
}

// Generate a new log file name based on the current timestamp
std::filesystem::path GPUDataCapture::GenerateLogFileName() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << cacheDirectory << "\\GPUData_" << time << ".log";
    return fs::path(ss.str());
}

// Get the current size of the cache directory
size_t GPUDataCapture::GetCurrentCacheSize() {
    size_t totalSize = 0;
    for (const auto& entry : fs::directory_iterator(cacheDirectory)) {
        if (fs::is_regular_file(entry)) {
            totalSize += fs::file_size(entry);
        }
    }
    return totalSize;
}
