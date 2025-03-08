#include "ArmageddonAlgorithm.h"
#include "CUDA_SPY.h" 
#include "SupervisorData.h"
#include "AI_Integration.h"  
#include <curl/curl.h>
#include <sstream>
#include <nlohmann/json.hpp>  
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <windows.h>
#include <psapi.h>

using json = nlohmann::json;

// **Constructor**
ArmageddonAlgorithm::ArmageddonAlgorithm() {
    std::cout << "ArmageddonAlgorithm initialized.\n";
}

// **Destructor**
ArmageddonAlgorithm::~ArmageddonAlgorithm() {
    std::cout << "ArmageddonAlgorithm destroyed.\n";
}

// **cURL Callback Function (Handles API Response)**
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// **Function to get CPU Usage across all cores**
double ArmageddonAlgorithm::GetCPUUsage() const {
    FILETIME idleTime, kernelTime, userTime;
    static ULARGE_INTEGER prevIdleTime, prevKernelTime, prevUserTime;

    if (!GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
        return -1.0; // Failed to get system times
    }

    ULARGE_INTEGER idle, kernel, user;
    idle.LowPart = idleTime.dwLowDateTime;
    idle.HighPart = idleTime.dwHighDateTime;
    kernel.LowPart = kernelTime.dwLowDateTime;
    kernel.HighPart = kernelTime.dwHighDateTime;
    user.LowPart = userTime.dwLowDateTime;
    user.HighPart = userTime.dwHighDateTime;

    ULONGLONG idleDiff = idle.QuadPart - prevIdleTime.QuadPart;
    ULONGLONG kernelDiff = kernel.QuadPart - prevKernelTime.QuadPart;
    ULONGLONG userDiff = user.QuadPart - prevUserTime.QuadPart;

    prevIdleTime = idle;
    prevKernelTime = kernel;
    prevUserTime = user;

    if (kernelDiff + userDiff == 0) return 0.0;

    return 100.0 - (idleDiff * 100.0 / (kernelDiff + userDiff));
}



// **Launch Ollama AI**
void LaunchOllama() {
    const char* appPath = "C:\\Users\\KrunkCiti\\AppData\\Local\\Programs\\Ollama\\ollama.exe";
    ShellExecute(NULL, "open", appPath, "run qwen2-math:7b", NULL, SW_SHOWNORMAL);
}

// **AI Query Function**
std::string QueryOllamaAI(int threadID, float computationResult, float predictedResult, float lookaheadPrediction, double cpuUsage) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    json requestJson = {
        {"model", "qwen2-math:7b"},
        {"prompt",
            "Thread ID: " + std::to_string(threadID) + "\n"
            "Computation Result: " + std::to_string(computationResult) + "\n"
            "Previous AI Prediction: " + std::to_string(predictedResult) + "\n"
            "Lookahead Prediction: " + std::to_string(lookaheadPrediction) + "\n"
            "Current CPU Usage: " + std::to_string(cpuUsage) + "%\n"
            "Analyze system performance and adjust predictions to maintain optimal CPU performance."
            "Generate insights on how cudaSpyData and supervisorData can optimize their workload to avoid CPU overload."
        }
    };

    std::string jsonData = requestJson.dump();
    std::cout << "Sending JSON: " << jsonData << std::endl;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());

        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20L);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        else {
            std::cout << "Raw AI Response: " << readBuffer << std::endl;
        }
        curl_easy_cleanup(curl);
    }
    else {
        std::cerr << "curl_easy_init() failed! Check cURL installation." << std::endl;
    }

    return readBuffer;
}

// **AI Learning Mechanism**
void ArmageddonAlgorithm::LearnFromCPUUsage(double cpuUsage, std::vector<float>& cudaSpyData, std::vector<float>& supervisorData) {
    if (cpuUsage > 20.0) {
        std::cout << "[AI ALERT] High CPU Usage Detected: " << cpuUsage << "%.\n";
        std::cout << "[AI RESPONSE] Optimizing workload balance between CUDA Spy & Supervisor.\n";

        for (size_t i = 0; i < cudaSpyData.size(); ++i) {
            cudaSpyData[i] *= 0.95f;
            supervisorData[i] *= 0.95f;
        }
    }
    else if (cpuUsage < 10.0) {
        std::cout << "[AI NOTE] CPU Usage is low: " << cpuUsage << "%.\n";
        std::cout << "[AI RESPONSE] Increasing processing efficiency.\n";

        for (size_t i = 0; i < cudaSpyData.size(); ++i) {
            cudaSpyData[i] *= 1.20f;
            supervisorData[i] *= 1.20f;
        }
    }
}

// **Calculate Prediction Using Formula**
float ArmageddonAlgorithm::CalculatePrediction(float computationResult, float predictedResult, float lookaheadPrediction) const {
    float newPrediction = (0.6f * computationResult)
        + (0.3f * predictedResult)
        + (0.1f * lookaheadPrediction);
    std::cout << "Calculated Prediction: " << newPrediction << std::endl;
    return newPrediction;
}

// **Function to Get System Memory Usage**
double ArmageddonAlgorithm::GetMemoryUsage() const {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        double totalMemory = static_cast<double>(memInfo.ullTotalPhys) / (1024 * 1024);  // Convert to MB
        double usedMemory = totalMemory - (static_cast<double>(memInfo.ullAvailPhys) / (1024 * 1024));
        return (usedMemory / totalMemory) * 100.0;  // Percentage of used memory
    }
    return -1.0;
}

// **Function to Get GPU Usage (NVIDIA-Specific)**
double ArmageddonAlgorithm::GetGPUUsage() const {
    HANDLE hProcess = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
        SIZE_T gpuMemoryUsed = pmc.WorkingSetSize;
        return static_cast<double>(gpuMemoryUsed) / (1024 * 1024);  // Convert to MB
    }
    return -1.0;
}



// **Core function for continuous learning**
void ArmageddonAlgorithm::ExecuteAlgorithm(std::vector<float>& cudaSpyData, std::vector<float>& supervisorData) {
    if (cudaSpyData.size() != supervisorData.size()) {
        std::cerr << "Error: Data size mismatch between CUDA Spy and Supervisor data.\n";
        return;
    }

    predictionData.cudaSpyData = cudaSpyData;
    predictionData.supervisorData = supervisorData;
    predictionData.outputData.resize(cudaSpyData.size());
    predictionData.outputData.resize(supervisorData.size());

    auto startTime = std::chrono::steady_clock::now();
    auto lastExecutionTime = std::chrono::steady_clock::now();

    while (true) {
        double cpuUsage = GetCPUUsage();
        auto currentTime = std::chrono::steady_clock::now();

        // Total elapsed time since we started
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - startTime
        ).count();

        // Time since last AI prediction
        auto timeSinceLastExecution = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastExecutionTime
        ).count();

        // Let CUDA and Supervisor run for the first 2 ms without AI interference
        if (elapsedTime > 2) {
            LearnFromCPUUsage(cpuUsage, cudaSpyData, supervisorData);
        }

        // **Execute AI Predictions every 3 minutes**
        if (timeSinceLastExecution >= 180000) {  // 180000 ms = 3 minutes
            std::cout << "[INFO] Running AI Predictions at "
                << (elapsedTime / 60000) << " minute(s)...\n";


            for (size_t i = 0; i < cudaSpyData.size(); ++i) {
                std::string aiResponse = QueryOllamaAI(
                    static_cast<int>(i),
                    cudaSpyData[i],
                    supervisorData[i],
                    0.0f,
                    cpuUsage
                );

                float aiPrediction = 0.0f;
                try {
                    aiPrediction = std::stof(aiResponse);
                    std::cout << "Extracted AI Prediction[" << i << "]: " << aiPrediction << std::endl;

                    // Update original data with AI Prediction
                    cudaSpyData[i] = aiPrediction;
                    supervisorData[i] = aiPrediction;
                }
                catch (const std::exception& e) {
                    std::cerr << "AI Response Parsing Error: " << e.what()
                        << " | Response: " << aiResponse << std::endl;
                }

                // Store in the output vector
                predictionData.outputData[i] = aiPrediction;
            }

            // Reset the timer for the next AI execution
            lastExecutionTime = std::chrono::steady_clock::now();
        }

        // Sleep for 1 second so we don’t eat CPU in this loop
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// **Setter for CUDA Spy Data (Ensuring Proper Update & Tracking Changes)**
void ArmageddonAlgorithm::SetCudaSpyData(const std::vector<float>& data) {
    if (data.size() != predictionData.cudaSpyData.size()) {
        std::cout << "[SETTER] CUDA Spy Data size changed from "
            << predictionData.cudaSpyData.size() << " to " << data.size() << "\n";
    }

    predictionData.cudaSpyData = data;
    predictionData.outputData.resize(data.size());  // Ensure outputData matches the new size
    std::cout << "[SETTER] CUDA Spy Data updated. Size: " << data.size() << "\n";
}

// **Setter for Supervisor Data (Tracking Size Changes)**
void ArmageddonAlgorithm::SetSupervisorData(const std::vector<float>& data) {
    if (data.size() != predictionData.supervisorData.size()) {
        std::cout << "[SETTER] Supervisor Data size changed from "
            << predictionData.supervisorData.size() << " to " << data.size() << "\n";
    }

    predictionData.supervisorData = data;
    predictionData.outputData.resize(data.size());
    std::cout << "[SETTER] Supervisor Data updated. Size: " << data.size() << "\n";
}

// **Getter for CUDA Spy Data**
const std::vector<float>& ArmageddonAlgorithm::GetCudaSpyData() const {
    return predictionData.cudaSpyData;
}

// **Getter for Supervisor Data**
const std::vector<float>& ArmageddonAlgorithm::GetSupervisorData() const {
    return predictionData.supervisorData;
}

// **Getter for Output Data (Including Warning if Data is Empty)**
const std::vector<float>& ArmageddonAlgorithm::GetOutputData() const {
    if (predictionData.outputData.empty()) {
        std::cerr << "[WARNING] Output data is empty! Ensure predictions have been made.\n";
    }
    return predictionData.outputData;
}

// **Print results & overwrite original data (Including Efficiency Score Calculation)**
void ArmageddonAlgorithm::PrintResults() {
    if (predictionData.outputData.empty()) {
        std::cerr << "[WARNING] No output data available! Ensure predictions have been executed.\n";
        return;
    }

    double efficiencyScore = 100.0;
    static double lastEfficiencyScore = 100.0;

    std::cout << "\n[AI Prediction Results]\n";
    for (size_t i = 0; i < predictionData.outputData.size(); ++i) {
        float predictedValue = predictionData.outputData[i];

        // **Calculate Efficiency Score Based on Change in Predictions**
        if (i < predictionData.cudaSpyData.size() && i < predictionData.supervisorData.size()) {
            float diff = std::abs(predictedValue - predictionData.cudaSpyData[i])
                + std::abs(predictedValue - predictionData.supervisorData[i]);

            efficiencyScore -= (diff * 2.5); // Reduces efficiency score based on deviation
        }

        if (efficiencyScore < 0.0) efficiencyScore = 0.0;

        // Overwrite the original CUDA Spy and Supervisor Data with AI Prediction
        predictionData.cudaSpyData[i] = predictedValue;
        predictionData.supervisorData[i] = predictedValue;

        std::cout << "Thread[" << i << "] - AI Prediction: " << predictedValue
            << " | Overwriting Original Data\n";
    }

    // **Display Efficiency Trend**
    std::cout << "[EFFICIENCY SCORE] " << efficiencyScore << "% (Previous: "
        << lastEfficiencyScore << "%)\n";

    lastEfficiencyScore = efficiencyScore;  // Store last recorded efficiency score

    std::cout << "[INFO] Original data successfully updated with AI Predictions.\n";
}

