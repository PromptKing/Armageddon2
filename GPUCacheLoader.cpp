// GPUCacheLoader.cpp
#include "GPUCacheLoader.h"

GPUCacheLoader::GPUCacheLoader(const std::string& cacheDirectory)
    : cacheDirectory(cacheDirectory) {
}

GPUCacheLoader::~GPUCacheLoader() {}

bool GPUCacheLoader::LoadCache() {
    try {
        for (const auto& entry : fs::directory_iterator(cacheDirectory)) {
            if (entry.is_regular_file()) {
                if (!LoadFile(entry.path().string())) {
                    std::cerr << "Failed to load file: " << entry.path() << std::endl;
                    return false;
                }
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error accessing cache directory: " << e.what() << std::endl;
        return false;
    }
}

const std::vector<std::vector<char>>& GPUCacheLoader::GetCacheData() const {
    return cacheData;
}

bool GPUCacheLoader::LoadFile(const std::string& filePath) {
    try {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filePath << std::endl;
            return false;
        }

        std::vector<char> fileData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        cacheData.push_back(std::move(fileData));
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading file: " << filePath << " - " << e.what() << std::endl;
        return false;
    }
}
