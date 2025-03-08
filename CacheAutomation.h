// CacheAutomation.h
#ifndef CACHE_AUTOMATION_H
#define CACHE_AUTOMATION_H

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <windows.h>
#include <cuda_runtime.h>
#include <map>

struct PredictionData {
    float computationResult;
    float predictedResult;
    std::vector<float> history;
    size_t historySize;
    float confidenceLevel; // Added to track prediction confidence
};

class CacheAutomation {
public:
    CacheAutomation(size_t cacheSize = 500);
    ~CacheAutomation();

    bool Initialize();
    void PerformAutomationRoutine(float input);
    void CachePredictionResult(const std::string& key, const PredictionData& data);
    bool RetrievePredictionResult(const std::string& key, PredictionData& data);

    // New database-related methods
    bool ConnectToDatabase(const std::string& databasePath);
    void StorePredictionToDatabase(const std::string& key, const PredictionData& data);
    bool LoadPredictionFromDatabase(const std::string& key, PredictionData& data);

    // New prediction evaluation method
    bool EvaluatePredictionOverride(float input, float prediction, float confidence); // Evaluate overriding CUDA_SPY

private:
    std::unordered_map<std::string, PredictionData> cache;
    std::mutex cacheMutex;
    size_t cacheSize;

    std::map<std::string, PredictionData> database; // Simulated database for storing predictions

    bool InitializeDatabaseConnection(); // Added method to initialize database connection
};

extern "C" __declspec(dllexport) void ExecuteAutomationRoutine(float input);

#endif // CACHE_AUTOMATION_H
