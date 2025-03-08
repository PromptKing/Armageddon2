#include "HyperCache.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <Windows.h> // For Windows API

// Constructor
HyperCache::HyperCache(const std::string& directory)
    : cacheDirectory(directory), cacheFileIndex(0) {
}

// Destructor
HyperCache::~HyperCache() {}

// Initialize the cache system
bool HyperCache::InitializeCache() {
    try {
        if (!DirectoryExists(cacheDirectory)) {
            if (!CreateDirectoryA(cacheDirectory.c_str(), nullptr)) {
                std::cerr << "Error creating directory: " << GetLastError() << std::endl;
                return false;
            }
        }
        return CreateCacheFile();
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing cache: " << e.what() << std::endl;
        return false;
    }
}

// Create a new cache file
bool HyperCache::CreateCacheFile() {
    try {
        currentCacheFile = GenerateCacheFileName();
        std::ofstream cacheFile(currentCacheFile, std::ios::out);
        if (cacheFile.is_open()) {
            cacheFile << "Cache initialized at: "
                << std::chrono::system_clock::now().time_since_epoch().count() << "\n";
            cacheFile.close();
            cacheFileIndex++;
            return true;
        }
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error creating cache file: " << e.what() << std::endl;
        return false;
    }
}

// Read all cache files
bool HyperCache::ReadCacheFiles() {
    try {
        WIN32_FIND_DATAA findFileData;
        HANDLE hFind = FindFirstFileA((cacheDirectory + "\\*.armcache").c_str(), &findFileData);

        if (hFind == INVALID_HANDLE_VALUE) {
            std::cerr << "No cache files found or error finding files. Error: " << GetLastError() << std::endl;
            return false;
        }

        do {
            std::string filePath = cacheDirectory + "\\" + findFileData.cFileName;
            std::ifstream cacheFile(filePath);
            if (cacheFile.is_open()) {
                std::string line;
                while (std::getline(cacheFile, line)) {
                    std::cout << line << std::endl;
                }
                cacheFile.close();
            }
        } while (FindNextFileA(hFind, &findFileData) != 0);

        FindClose(hFind);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading cache files: " << e.what() << std::endl;
        return false;
    }
}

// Generate cache file name
std::string HyperCache::GenerateCacheFileName() const {
    return cacheDirectory + "\\cache_" + std::to_string(cacheFileIndex) + ".armcache";
}

// Save runtime data to the current cache
bool HyperCache::SaveDataToCache(const std::vector<std::string>& data) {
    try {
        std::ofstream cacheFile(currentCacheFile, std::ios::app);
        if (cacheFile.is_open()) {
            for (const auto& entry : data) {
                cacheFile << entry << "\n";
            }
            cacheFile.close();
            return true;
        }
        return false; // File could not be opened
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving data to cache: " << e.what() << std::endl;
        return false;
    }
}

// Helper function to check if a directory exists
bool HyperCache::DirectoryExists(const std::string& directory) const {
    DWORD fileAttributes = GetFileAttributesA(directory.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES &&
        (fileAttributes & FILE_ATTRIBUTE_DIRECTORY));
}
