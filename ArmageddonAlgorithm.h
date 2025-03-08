#ifndef ARMAGEDDON_ALGORITHM_H
#define ARMAGEDDON_ALGORITHM_H

#include "CUDA_SPY.h"
#include "SupervisorData.h"
#include <vector>
#include <string>
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <windows.h>
#include <curl/curl.h>
#include <immintrin.h> // For AVX instructions


// **Structure for handling predictions**
struct PredictionData {
    std::vector<float> cudaSpyData;
    std::vector<float> supervisorData;
    std::vector<float> outputData;
};

// **Class Definition**
class ArmageddonAlgorithm {
public:
    // **Constructor and Destructor**
    ArmageddonAlgorithm();
    ~ArmageddonAlgorithm();

    // **Core Execution Function**
    void ExecuteAlgorithm(std::vector<float>& cudaSpyData, std::vector<float>& supervisorData);

    // **Apply AI-generated kernel optimizations**
    void ApplyAIKernelOptimization(const std::string& aiMathFunction,
        std::vector<float>& cudaSpyData,
        std::vector<float>& supervisorData,
        size_t index);

    // **Process AI-generated mathematical logic dynamically**
    float ProcessAIMathFunction(const std::string& aiMathFunction);

    // **Apply AI-generated AVX Optimizations**
    void ApplyAIAVXOptimization(const std::string& avxInstructions);

    // **Print results & overwrite original data**
    void PrintResults();

    // **Setters for prediction data**
    void SetCudaSpyData(const std::vector<float>& data);
    void SetSupervisorData(const std::vector<float>& data);

    // **Getters for prediction data**
    const std::vector<float>& GetCudaSpyData() const;
    const std::vector<float>& GetSupervisorData() const;
    const std::vector<float>& GetOutputData() const;

    // **Apply AVX-enhanced calculations**
    void ApplyAVXOptimization(float newPrediction);

private:
    PredictionData predictionData;

    // **AI Prediction Calculation**
    float CalculatePrediction(float computationResult, float predictedResult, float lookaheadPrediction) const;

    // **System Monitoring Functions**
    double GetCPUUsage() const;
    double GetMemoryUsage() const;
    double GetGPUUsage() const;

    // **AI Learning Mechanism**
    void LearnFromCPUUsage(double cpuUsage, std::vector<float>& cudaSpyData, std::vector<float>& supervisorData);
    void LearnFromGPUUsage(double gpuUsage, std::vector<float>& cudaSpyData, std::vector<float>& supervisorData);
    void LearnFromMemoryUsage(double memoryUsage, std::vector<float>& cudaSpyData, std::vector<float>& supervisorData);

    // **Timer Mechanism**
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point lastExecutionTime;

    // **AVX-Enhanced Computation Storage**
#ifdef _MSC_VER  // MSVC Compiler
    __declspec(align(32)) float computationHistory[8] = { 0.0f };
#else  // GCC/Clang
    float computationHistory[8] __attribute__((aligned(32))) = { 0.0f };
#endif

    bool avxOptimized = false;
    float computationResult = 0.0f;
};

// **AI Query Function**

std::string QueryOllamaAI(int threadID, float computationResult, float predictedResult,
    float lookaheadPrediction, double cpuUsage, double memoryUsage,
    double gpuUsage, bool avxOptimized,
    __m256 originalData, __m256 predictionData, __m256 updatedData);

// **Launch Ollama AI**
void LaunchOllama();

// **cURL Callback Function (Handles API Response)**
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);

#endif  // ARMAGEDDON_ALGORITHM_H
