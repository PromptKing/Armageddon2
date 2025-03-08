#ifndef HYPERCACHE_H
#define HYPERCACHE_H

#include <string>
#include <vector>

class HyperCache {
public:
    /**
     * @brief Constructor to set up the cache directory.
     * @param directory Directory where the cache files will be stored.
     */
    explicit HyperCache(const std::string& directory = "F:\\Armageddon Cache");

    /**
     * @brief Destructor to clean up resources.
     */
    ~HyperCache();

    /**
     * @brief Initialize the cache system.
     * Creates the cache directory if it doesn't exist and sets up the initial cache file.
     * @return True if successful, false otherwise.
     */
    bool InitializeCache();

    /**
     * @brief Create a new cache file.
     * Generates a new cache file for storing data.
     * @return True if successful, false otherwise.
     */
    bool CreateCacheFile();

    /**
     * @brief Read all cache files in the cache directory.
     * Iterates through `.armcache` files in the directory and reads their contents.
     * @return True if successful, false otherwise.
     */
    bool ReadCacheFiles();

    /**
     * @brief Save runtime data to the current cache file.
     * Appends the given data to the current cache file.
     * @param data A vector of strings representing the data to be written.
     * @return True if successful, false otherwise.
     */
    bool SaveDataToCache(const std::vector<std::string>& data);

private:
    std::string cacheDirectory;     // Directory where cache files are stored
    std::string currentCacheFile;   // Current cache file being written to
    size_t cacheFileIndex;          // Index for naming cache files

    /**
     * @brief Generate a new cache file name.
     * Constructs the file name for a new cache file based on the cache directory and index.
     * @return A string representing the generated cache file name.
     */
    std::string GenerateCacheFileName() const;

    /**
     * @brief Check if a directory exists.
     * Verifies whether the specified directory exists in the file system.
     * @param directory The path to the directory.
     * @return True if the directory exists, false otherwise.
     */
    bool DirectoryExists(const std::string& directory) const;
};

#endif // HYPERCACHE_H
