#pragma once
#ifndef ARMAGEDDON_ALGORITHM_H
#define ARMAGEDDON_ALGORITHM_H

#include <vector>
#include <string>

class ArmageddonAlgorithm {
public:
    // Constructor and Destructor
    ArmageddonAlgorithm();
    ~ArmageddonAlgorithm();

    // Core Algorithm Execution
    void ExecuteAlgorithm(std::vector<float>& cudaSpyData, std::vector<float>& supervisorData);
    void LearnFromCPUUsage(double cpuUsage, std::vector<float>& cudaSpyData, std::vector<float>& supervisorData);
    float CalculatePrediction(float computationResult, float predictedResult, float lookaheadPrediction) const;

    // Data Setters and Getters
    void SetCudaSpyData(const std::vector<float>& data);
    void SetSupervisorData(const std::vector<float>& data);
    const std::vector<float>& GetCudaSpyData() const;
    const std::vector<float>& GetSupervisorData() const;
    const std::vector<float>& GetOutputData() const;

    // Performance Monitoring
    double GetCPUUsage() const;
    void PrintResults();

private:
    // Data storage structure
    struct PredictionData {
        std::vector<float> cudaSpyData;
        std::vector<float> supervisorData;
        std::vector<float> outputData;
    } predictionData;

    // AI Query Handling
    std::string QueryOllamaAI(int threadID, float computationResult, float predictedResult, float lookaheadPrediction, double cpuUsage);
};

// External Functions
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
void LaunchOllama();

#endif // ARMAGEDDON_ALGORITHM_H
