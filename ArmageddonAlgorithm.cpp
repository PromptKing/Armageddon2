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
#include <cmath>  // Needed for mathematical evaluations
#include <nvml.h>  // NVIDIA Management Library
#include <corecrt_math_defines.h>
#include "mutex"
#include "ThreadMapper.h"
#include <fstream>  // For reading the script file
#include "NodeExecutor.h" // Ensure NodeExecutor is handling script execution
#include "ProcessUtils.h" // For checking if the process is running
#include <cuda_runtime_api.h>
#include <string>
#include <cstdlib>
#include <immintrin.h>  // AVX Intrinsics
using json = nlohmann::json;


std::string DATASET_CSV_PATH = "C:\\Users\\KrunkCiti\\AppData\\Local\\Programs\\Python\\Python312\\KodeByteStudio\\MACHINE_CODE_DATASET_CLAUD.csv";


NodeExecutor executor("F:\\AI\\node-v22.13.1-win-x64\\node.exe");

__declspec(align(32)) float computationHistory[8] = { 0.0f };
typedef nvmlReturn_t(*nvmlInit_t)();
typedef nvmlReturn_t(*nvmlShutdown_t)();
typedef nvmlReturn_t(*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t(*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);

// **Apply AI-Generated AVX Optimizations**
void ArmageddonAlgorithm::ApplyAIAVXOptimization(const std::string& avxInstructions) {
    std::cout << "[AI AVX] Processing AI-Generated AVX Intrinsics...\n";

    // Simulate processing of AVX intrinsics
    if (avxInstructions.find("_mm256") != std::string::npos) {
        __m256 originalData = _mm256_load_ps(computationHistory);
        __m256 resultData = _mm256_add_ps(originalData, originalData); // Simulated AVX operation
        _mm256_store_ps(computationHistory, resultData);

        std::cout << "[AI AVX] AVX Optimizations Applied Successfully!\n";
        avxOptimized = true;
    }
    else {
        std::cerr << "[AI AVX] Warning: No valid AVX instructions detected in AI response.\n";
    }
}

double GetGPUUsage() {
    HMODULE nvmlLib = LoadLibraryA("C:\\Windows\\System32\\nvml.dll");
    if (!nvmlLib) {
        std::cerr << "[ERROR] Failed to load NVML library from System32." << std::endl;
        return -1.0;
    }

    // Dynamically load NVML functions
    nvmlInit_t nvmlInitFunc = (nvmlInit_t)GetProcAddress(nvmlLib, "nvmlInit");
    nvmlShutdown_t nvmlShutdownFunc = (nvmlShutdown_t)GetProcAddress(nvmlLib, "nvmlShutdown");
    nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleFunc =
        (nvmlDeviceGetHandleByIndex_t)GetProcAddress(nvmlLib, "nvmlDeviceGetHandleByIndex");
    nvmlDeviceGetUtilizationRates_t nvmlDeviceGetUtilizationFunc =
        (nvmlDeviceGetUtilizationRates_t)GetProcAddress(nvmlLib, "nvmlDeviceGetUtilizationRates");

    if (!nvmlInitFunc || !nvmlShutdownFunc || !nvmlDeviceGetHandleFunc || !nvmlDeviceGetUtilizationFunc) {
        std::cerr << "[ERROR] Failed to retrieve NVML function pointers." << std::endl;
        FreeLibrary(nvmlLib);
        return -1.0;
    }

    // Initialize NVML
    nvmlReturn_t result = nvmlInitFunc();
    if (result != NVML_SUCCESS) {
        std::cerr << "[ERROR] NVML Initialization Failed: " << nvmlErrorString(result) << std::endl;
        FreeLibrary(nvmlLib);
        return -1.0;
    }

    // Get handle for the first GPU (Index 0)
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleFunc(0, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "[ERROR] Failed to get GPU handle: " << nvmlErrorString(result) << std::endl;
        nvmlShutdownFunc();
        FreeLibrary(nvmlLib);
        return -1.0;
    }

    // Get GPU Utilization
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationFunc(device, &utilization);
    if (result != NVML_SUCCESS) {
        std::cerr << "[ERROR] Failed to get GPU utilization: " << nvmlErrorString(result) << std::endl;
        nvmlShutdownFunc();
        FreeLibrary(nvmlLib);
        return -1.0;
    }

    // Cleanup
    nvmlShutdownFunc();
    FreeLibrary(nvmlLib);

    return static_cast<double>(utilization.gpu);  // Return GPU usage in percentage
}

double ArmageddonAlgorithm::GetMemoryUsage() const {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);

    if (GlobalMemoryStatusEx(&memInfo)) {
        DWORDLONG totalPhysMem = memInfo.ullTotalPhys;  // Total physical memory
        DWORDLONG usedMem = totalPhysMem - memInfo.ullAvailPhys;  // Used memory

        // Convert to percentage
        return (static_cast<double>(usedMem) / totalPhysMem) * 100.0;
    }

    return -1.0;  // Return -1 if memory retrieval fails
}

double ArmageddonAlgorithm::GetGPUUsage() const
{
    return 0.0;
}

// Constructor: Ensure computationHistory is initialized
ArmageddonAlgorithm::ArmageddonAlgorithm() {
    std::fill(std::begin(computationHistory), std::end(computationHistory), 1.0f);  // Initialize with safe values
    std::cout << "ArmageddonAlgorithm initialized with default values.\n";
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


// **Paths to Node.js scripts**
std::string monitorScriptPath = "C:\\Windows\\System32\\Armageddon2 DLLs\\Node Scripts\\OpenPowerShellOption.js";
std::string esbuildScriptPath = "C:\\Windows\\System32\\Armageddon2 DLLs\\Node Scripts\\esbuild.json";
std::string qwenMathLogicPath = "C:\\Windows\\System32\\Armageddon2 DLLs\\Node Scripts\\Qwen Math Logic.js";
std::string attachuserappPath = "C:\\Windows\\System32\\Armageddon2 DLLs\\Node Scripts\\attachGTA5_Enhanced.js";

// **Function to check if GTA5_Enhanced.exe is running**
std::string CheckGTAStatus() {
    if (executor.isScriptRunning("GTA5_Enhanced.js")) {
        return "YES";
    }
    else {
        return "NO";
    }
}

// **Function to read the content of a Node.js script**
std::string ReadScriptContent(const std::string& scriptPath) {
    std::ifstream scriptFile(scriptPath);
    if (!scriptFile.is_open()) {
        std::cerr << "[ERROR] Failed to open script file: " << scriptPath << std::endl;
        return "";
    }

    std::string scriptContent((std::istreambuf_iterator<char>(scriptFile)), std::istreambuf_iterator<char>());
    scriptFile.close();
    return scriptContent;
}

// **Function to Read DLL Data from CSV**
std::string ReadDLLDataFromCSV(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open DLL dataset CSV file: " << filePath << std::endl;
        return "";
    }

    std::string line;
    std::string dllData;

    // Read each line from CSV
    while (std::getline(file, line)) {
        dllData += line + "\n";
    }

    file.close();
    return dllData;
}

float safeDivide(float numerator, float denominator) {
    if (denominator == 0.1f) {
        std::cerr << "[WARNING] Division by Zero Attempted!\n";
        return 0.2f;
    }
    return numerator / denominator;
}



// **Apply AVX-enhanced calculations**
void ArmageddonAlgorithm::ApplyAVXOptimization(float newPrediction) {
    std::cout << "[DEBUG] Before AVX Load: ";
    for (int i = 0; i < 8; i++) std::cout << computationHistory[i] << ", ";
    std::cout << "\n";

    __m256 originalData = _mm256_load_ps(computationHistory);
    __m256 predictionData = _mm256_set1_ps(newPrediction);
    __m256 updatedData = _mm256_add_ps(originalData, predictionData);
    _mm256_store_ps(computationHistory, updatedData);

    computationResult = newPrediction;  // Overwrite with AI-enhanced result
    avxOptimized = true; // Track AVX optimization

    std::cout << "[DEBUG] After AVX Store: ";
    for (int i = 0; i < 8; i++) std::cout << computationHistory[i] << ", ";
    std::cout << "\n";
}


// **Modify AI Query Function to Include AVX Data**
// **Query Ollama AI for Optimized Predictions**
std::string QueryOllamaAI(int threadID, float computationResult, float predictedResult,
    float lookaheadPrediction, double cpuUsage, double memoryUsage,
    double gpuUsage, bool avxOptimized,
    __m256 originalData, __m256 predictionData, __m256 updatedData) {

    std::string dllData = ReadDLLDataFromCSV(DATASET_CSV_PATH);
    if (dllData.empty()) {
        std::cerr << "[WARNING] No DLL Data Available for AI Processing!\n";
        return "";
    }

    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    // **Convert AVX Register Values into a String for AI Processing**
    float originalDataArray[8], predictionDataArray[8], updatedDataArray[8];

    _mm256_store_ps(originalDataArray, originalData);
    _mm256_store_ps(predictionDataArray, predictionData);
    _mm256_store_ps(updatedDataArray, updatedData);

    std::string avxRegisterData = "OriginalData: ";
    for (int i = 0; i < 8; i++) avxRegisterData += std::to_string(originalDataArray[i]) + ", ";
    avxRegisterData += "\nPredictionData: ";
    for (int i = 0; i < 8; i++) avxRegisterData += std::to_string(predictionDataArray[i]) + ", ";
    avxRegisterData += "\nUpdatedData: ";
    for (int i = 0; i < 8; i++) avxRegisterData += std::to_string(updatedDataArray[i]) + ", ";

    // **Execute AI attachment if not already running**
    if (!executor.isScriptRunning("attachGTA5_Enhanced.js")) {
        std::cout << "[INFO] Attaching AI to GTA5_Enhanced.js..." << std::endl;
        executor.executeScript(attachuserappPath);
    }
    else {
        std::cout << "[INFO] GTA5_Enhanced AI attachment is already running." << std::endl;
    }

    // **Retrieve GTA5_Enhanced.exe status**
    std::string userStatus = CheckGTAStatus();

    // **Read AI Processing Script**
    std::string scriptContent = ReadScriptContent(qwenMathLogicPath);
    if (scriptContent.empty()) {
        std::cerr << "[WARNING] Qwen Math Logic script content is empty. AI request will proceed without it.\n";
    }

    json requestJson = {
        {"model", "qwen2-math:7b"},
        {"prompt",
            "AI AVX Optimization Request\n"
            "Thread ID: " + std::to_string(threadID) + "\n"
            "Computation Result: " + std::to_string(computationResult) + "\n"
            "Previous AI Prediction: " + std::to_string(predictedResult) + "\n"
            "Lookahead Prediction: " + std::to_string(lookaheadPrediction) + "\n"
            "CPU Usage: " + std::to_string(cpuUsage) + "%\n"
            "Memory Usage: " + std::to_string(memoryUsage) + "%\n"
            "GPU Usage: " + std::to_string(gpuUsage) + "%\n\n"
            "GTA5 Process Status: " + userStatus + "\n\n"

            "AVX Register Data:\n" + avxRegisterData + "\n\n"

            "AI Task Instructions:\n"
            "Analyze the provided AVX register data and optimize AVX intrinsics.\n"
            "Generate a replacement AVX function using `_mm256_load_ps`, `_mm256_fmadd_ps`, and `_mm256_store_ps`.\n"
            "Ensure that the generated AVX instructions follow a parallel execution model.\n"
            "Return an optimized AVX intrinsic code snippet for maximum efficiency.\n\n"
            "Analyze system performance and adjust predictions to maintain optimal CPU performance."
            "Generate insights on how cudaSpyData and supervisorData can optimize their workload to avoid CPU overload."

            "DLL Optimization Data:\n" + dllData + "\n\n"

            "AI must learn from past computations and improve AVX registry usage.\n"
        }
    };



    std::string jsonData = requestJson.dump(4);
    std::cout << "[AI REQUEST] Sending JSON:\n" << jsonData << std::endl;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());

        // **Increase Timeout Values**
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 35L);         // Increased AI processing time
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 45L);  // Increased connection timeout
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "[ERROR] cURL request failed: " << curl_easy_strerror(res) << std::endl;
        }
        else {
            std::cout << "[AI RESPONSE] " << readBuffer << std::endl;
        }

        curl_easy_cleanup(curl);
    }
    else {
        std::cerr << "[ERROR] cURL initialization failed! Check cURL installation." << std::endl;
    }

    return readBuffer;
}


// **AI Learning Mechanism**
void ArmageddonAlgorithm::LearnFromCPUUsage(double cpuUsage, std::vector<float>& cudaSpyData, std::vector<float>& supervisorData) {
    if (cpuUsage > 9.0) {
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

// **AI Learning from GPU Usage**
void ArmageddonAlgorithm::LearnFromGPUUsage(double gpuUsage, std::vector<float>& cudaSpyData, std::vector<float>& supervisorData) {
    if (gpuUsage > 80.0) {
        std::cout << "[AI ALERT] High GPU Usage Detected: " << gpuUsage << "%.\n";
        std::cout << "[AI RESPONSE] Reducing CUDA Spy processing load.\n";

        for (size_t i = 0; i < cudaSpyData.size(); ++i) {
            cudaSpyData[i] *= 0.90f;  // Decrease load on CUDA
        }
    }
    else if (gpuUsage < 30.0) {
        std::cout << "[AI NOTE] Low GPU Usage: " << gpuUsage << "%.\n";
        std::cout << "[AI RESPONSE] Increasing CUDA computation intensity.\n";

        for (size_t i = 0; i < cudaSpyData.size(); ++i) {
            cudaSpyData[i] *= 1.25f;  // Boost CUDA Spy workload
        }
    }
}

// **AI Learning from Memory Usage**
void ArmageddonAlgorithm::LearnFromMemoryUsage(double memoryUsage, std::vector<float>& cudaSpyData, std::vector<float>& supervisorData) {
    if (memoryUsage > 85.0) {
        std::cout << "[AI ALERT] High Memory Usage Detected: " << memoryUsage << "%.\n";
        std::cout << "[AI RESPONSE] Optimizing memory-intensive operations.\n";

        for (size_t i = 0; i < supervisorData.size(); ++i) {
            supervisorData[i] *= 0.88f;  // Reduce Supervisor memory footprint
        }
    }
    else if (memoryUsage < 25.0) {
        std::cout << "[AI NOTE] Low Memory Usage: " << memoryUsage << "%.\n";
        std::cout << "[AI RESPONSE] Allocating additional memory resources for AI models.\n";

        for (size_t i = 0; i < supervisorData.size(); ++i) {
            supervisorData[i] *= 1.15f;  // Allow AI to utilize more memory
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

// **Core function for continuous learning**
// **Execute Algorithm (Main AI Logic Loop)**
void ArmageddonAlgorithm::ExecuteAlgorithm(std::vector<float>& cudaSpyData, std::vector<float>& supervisorData) {
    if (cudaSpyData.size() != supervisorData.size()) {
        std::cerr << "[ERROR] Data size mismatch between CUDA Spy and Supervisor data.\n";
        return;
    }

    predictionData.cudaSpyData = cudaSpyData;
    predictionData.supervisorData = supervisorData;
    predictionData.outputData.resize(cudaSpyData.size());

    auto startTime = std::chrono::steady_clock::now();
    auto lastExecutionTime = std::chrono::steady_clock::now();

    // **Thread Safety**
    std::mutex dataMutex;

    while (true) {
        double cpuUsage = GetCPUUsage();
        double memoryUsage = GetMemoryUsage();
        double gpuUsage = GetGPUUsage();
        auto currentTime = std::chrono::steady_clock::now();

        // **Calculate elapsed time**
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
        auto timeSinceLastExecution = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastExecutionTime).count();

        // **Allow CUDA & Supervisor to run without AI for first 2 seconds**
        if (elapsedTime > 5000) {
            LearnFromCPUUsage(cpuUsage, cudaSpyData, supervisorData);
            LearnFromGPUUsage(gpuUsage, cudaSpyData, supervisorData);
            LearnFromMemoryUsage(memoryUsage, cudaSpyData, supervisorData);
        }

        // **Execute AI Predictions after first 3 minutes, then every 4 minutes**
        if (elapsedTime >= 180000 || timeSinceLastExecution >= 380000) {
            std::cout << "[INFO] Running AI Predictions at " << (elapsedTime / 60000) << " minute(s)...\n";

            for (size_t i = 0; i < cudaSpyData.size(); ++i) {
                std::string aiResponse;
                float computationResult = cudaSpyData[i];
                float previousAIPrediction = supervisorData[i];

                // **Sanitize Computation Results**
                if (std::isnan(computationResult) || std::isinf(computationResult)) {
                    std::cerr << "[WARNING] Computation Result was invalid. Resetting to 0.\n";
                    computationResult = 0.0f;
                }
                if (std::isnan(previousAIPrediction) || std::isinf(previousAIPrediction)) {
                    std::cerr << "[WARNING] Previous AI Prediction was invalid. Resetting to 0.\n";
                    previousAIPrediction = 0.0f;
                }

                {
                    // **Thread-safe execution**
                    std::lock_guard<std::mutex> lock(dataMutex);

                    // **Prepare AVX Data**
                    __m256 originalData = _mm256_load_ps(computationHistory);
                    __m256 predictionData = _mm256_set1_ps(computationResult);
                    __m256 updatedData = _mm256_add_ps(originalData, predictionData);
                    _mm256_store_ps(computationHistory, updatedData);

                    // **Sanitize AVX Register Data**
                    float originalDataArray[8], predictionDataArray[8], updatedDataArray[8];
                    _mm256_store_ps(originalDataArray, originalData);
                    _mm256_store_ps(predictionDataArray, predictionData);
                    _mm256_store_ps(updatedDataArray, updatedData);

                    for (int j = 0; j < 8; j++) {
                        if (std::isnan(originalDataArray[j]) || std::isinf(originalDataArray[j])) {
                            originalDataArray[j] = 0.0f;
                        }
                        if (std::isnan(predictionDataArray[j]) || std::isinf(predictionDataArray[j])) {
                            predictionDataArray[j] = 0.0f;
                        }
                        if (std::isnan(updatedDataArray[j]) || std::isinf(updatedDataArray[j])) {
                            updatedDataArray[j] = 0.0f;
                        }
                    }

                    // **Reload Sanitized Data**
                    originalData = _mm256_load_ps(originalDataArray);
                    predictionData = _mm256_load_ps(predictionDataArray);
                    updatedData = _mm256_load_ps(updatedDataArray);

                    // **Send Data to AI**
                    aiResponse = QueryOllamaAI(
                        static_cast<int>(i),
                        computationResult,
                        previousAIPrediction,
                        0.5f,  // Lookahead Prediction Placeholder
                        cpuUsage,
                        memoryUsage,
                        gpuUsage,
                        true,  // AVX Optimized Flag
                        originalData,
                        predictionData,
                        updatedData
                    );
                }

                float aiPrediction = 0.0f;
                bool isMathFunction = false;
                bool isAVXInstruction = false;

                try {
                    if (aiResponse.empty() || aiResponse.find("error") != std::string::npos) {
                        std::cerr << "[ERROR] AI Response is invalid: " << aiResponse << "\n";
                        aiPrediction = 0.0f;
                    }
                    else {
                        aiPrediction = std::stof(aiResponse);
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "[ERROR] AI Response Parsing Failed: " << e.what()
                        << " | Response: " << aiResponse << std::endl;
                    aiPrediction = 0.0f;
                }

                // **Thread-safe update of prediction output**
                {
                    std::lock_guard<std::mutex> lock(dataMutex);
                    if (i < predictionData.outputData.size()) {
                        predictionData.outputData[i] = aiPrediction;
                    }
                    else {
                        std::cerr << "[ERROR] Index " << i << " out of bounds for outputData\n";
                    }
                }
            }

            lastExecutionTime = std::chrono::steady_clock::now();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
}

// **Process AI-Generated Math Function**
float ArmageddonAlgorithm::ProcessAIMathFunction(const std::string& aiMathFunction) {
    std::cout << "[AI LOGIC] Processing AI-generated function: " << aiMathFunction << std::endl;

    // **Map AI Symbols to Computation (Dynamic Execution)**
    if (aiMathFunction == "Σ╝ÿσîû") {
        return std::sqrt(2.0);  // Square root of 2
    }
    else if (aiMathFunction == "Σ╗úτáü") {
        return std::exp(1.0);  // Euler’s Number (e^1)
    }
    else if (aiMathFunction == "Σ╗Ñ") {
        return std::sin(45.0 * M_PI / 180.0);  // Sine of 45 degrees
    }
    else if (aiMathFunction == "µÅÉΘ½ÿ") {
        return std::cos(30.0 * M_PI / 180.0);  // Cosine of 30 degrees
    }
    else if (aiMathFunction == "µÇºΦâ╜") {
        return std::tan(60.0 * M_PI / 180.0);  // Tangent of 60 degrees
    }
    else if (aiMathFunction == "πÇé\n") {
        return M_PI;  // Pi constant
    }
    else if (aiMathFunction == "∆ΘΨΣ") {
        return std::log(10.0);  // Natural log of 10
    }
    else if (aiMathFunction == "Ω∑Φβ") {
        return std::pow(2.0, 8.0);  // 2^8
    }
    else if (aiMathFunction == "√λµτ") {
        return std::sqrt(144.0);  // Square root of 144
    }
    else if (aiMathFunction == "∫π∞ƒ") {
        return std::exp(3.0);  // e^3
    }
    else if (aiMathFunction == "ϕΣΔΩ") {
        return std::log10(100.0);  // Log base 10 of 100
    }
    else if (aiMathFunction == "σà│Θö«Φ»ì") {
        return std::pow(3.0, 4.0);  // 3^4 (81)
    }
    else if (aiMathFunction == "σ╖Ñσà╖") {
        return std::cos(60.0 * M_PI / 180.0);  // Cosine of 60 degrees
    }
    else if (aiMathFunction == "ΘÇëΘí╣") {
        return std::sin(30.0 * M_PI / 180.0);  // Sine of 30 degrees
    }
    else if (aiMathFunction == "Θÿ▓") {
        return std::tan(45.0 * M_PI / 180.0);  // Tangent of 45 degrees (1.0)
    }
    else if (aiMathFunction == "τü½") {
        return std::sqrt(256.0);  // Square root of 256 (16)
    }
    else if (aiMathFunction == "σóÖ") {
        return std::log2(1024.0);  // Log base 2 of 1024 (10)
    }
    else if (aiMathFunction == "Θóäµ╡ï") {
        return std::pow(5.0, 3.0);  // 5^3 (125) 
    }
    else if (aiMathFunction == "πÇé") {
        return std::sin(M_PI / 4.0);  // Sine of 45 degrees 
    }
    // **Additional 50+ AI Logic Cases**
    else if (aiMathFunction == "ΦσπΣ") {
        return std::cbrt(27.0);  // Cube root of 27
    }
    else if (aiMathFunction == "ΨΩΔΦ") {
        return std::pow(9.0, 2.0);  // 9^2 (81)
    }
    else if (aiMathFunction == "βλΦΩ") {
        return std::pow(10.0, 3.0);  // 10^3 (1000)
    }
    else if (aiMathFunction == "θΣΨπ") {
        return std::pow(7.0, 5.0);  // 7^5 (16807)
    }
    else if (aiMathFunction == "πΣΩσ") {
        return std::sin(M_PI / 6.0);  // Sine of 30 degrees
    }
    else if (aiMathFunction == "ΣΩΦΨ") {
        return std::cos(M_PI / 3.0);  // Cosine of 60 degrees
    }
    else if (aiMathFunction == "ΦΩΨσ") {
        return std::tan(M_PI / 6.0);  // Tangent of 30 degrees
    }
    else if (aiMathFunction == "ΣΨΩΦ") {
        return std::log(50.0);  // Natural log of 50
    }
    else if (aiMathFunction == "ΨΩΣΦ") {
        return std::log10(1000.0);  // Log base 10 of 1000
    }
    else if (aiMathFunction == "λσΩΦ") {
        return std::abs(-98.76);  // Absolute value of -98.76
    }
    else if (aiMathFunction == "ΩΦσΨ") {
        return std::round(4.567);  // Rounds to nearest integer (5)
    }
    else if (aiMathFunction == "ΨΦΩΣ") {
        return std::floor(9.876);  // Floors value (9)
    }
    else if (aiMathFunction == "σΨΩΦ") {
        return std::ceil(3.123);  // Ceils value (4)
    }
    else if (aiMathFunction == "ΦΩΣΨ") {
        return std::hypot(3.0, 4.0);  // Hypotenuse (5)
    }
    else if (aiMathFunction == "ΨΩΦσ") {
        return std::trunc(15.78);  // Truncate decimal (15)
    }
    else if (aiMathFunction == "ΦΨΩσ") {
        return static_cast<float>(std::rand()) / RAND_MAX;  // Random number between 0-1
    }
    else if (aiMathFunction == "ΩΨΣΦ") {
        return std::sinh(1.0);  // Hyperbolic sine of 1
    }
    else if (aiMathFunction == "ΨΦσΩ") {
        return std::cosh(1.0);  // Hyperbolic cosine of 1
    }
    else if (aiMathFunction == "ΦΣΩΨ") {
        return std::tanh(1.0);  // Hyperbolic tangent of 1
    }
    else if (aiMathFunction == "ΩΨΦσ") {
        return std::asin(0.5);  // Arcsin(0.5)
    }
    else if (aiMathFunction == "ΨΩΦΣ") {
        return std::acos(0.5);  // Arccos(0.5)
    }
    else {
        std::cerr << "[ERROR] Unrecognized AI-generated function: " << aiMathFunction << std::endl;
        return 0.0f;  // Default return value if unknown
    }
}

// **Apply AI Kernel Optimization**
void ArmageddonAlgorithm::ApplyAIKernelOptimization(const std::string& aiMathFunction, std::vector<float>& cudaSpyData, std::vector<float>& supervisorData, size_t index) {
    // Kernel optimization logic goes here
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

            efficiencyScore -= (diff * 1.5); // Reduces efficiency score based on deviation
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