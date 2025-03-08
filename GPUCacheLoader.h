// GPUCacheLoader.h
#ifndef GPU_CACHE_LOADER_H
#define GPU_CACHE_LOADER_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

class GPUCacheLoader {
public:
    // Constructor with a default cache directory
    GPUCacheLoader(const std::string& cacheDirectory = "C:\\Windows\\System32\\Armageddon2 DLLs\\fe8d97be6d92aa79");
    ~GPUCacheLoader();

    // Loads cache files from the specified directory
    bool LoadCache();

    // Gets the loaded cache data
    const std::vector<std::vector<char>>& GetCacheData() const;

private:
    std::string cacheDirectory;
    std::vector<std::vector<char>> cacheData;

    // Helper function to read a file into memory
    bool LoadFile(const std::string& filePath);
};

#endif // GPU_CACHE_LOADER_H
