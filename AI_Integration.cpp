#include "AI_Integration.h"
#include <iostream>
#include <curl/curl.h>
#include <sstream>

// Callback function to handle API response
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Function to send thread data to AI and get a prediction
std::string QueryOllamaAI(int threadID, float computationResult, float predictedResult, float lookaheadPrediction, const std::string& modelPath) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;
    std::ostringstream jsonDataStream;

    // ✅ Include the full model path in the JSON request
    jsonDataStream << R"({
        "model": "C:\\Users\\KrunkCiti\\.ollama\\models\\blobs\\sha256-6340dc3229b0d08ea9cc49b75d4098702983e17b4c096d57afbbf2ffc813f2be",
        "prompt": "Thread ID: )" << threadID << R"(\n"
        "Computation Result: )" << computationResult << R"(\n"
        "Previous AI Prediction: )" << predictedResult << R"(\n"
        "Lookahead Prediction: )" << lookaheadPrediction << R"(\n"
        "Using model path: )" << modelPath << R"(\n"
        "Predict the next computation value accurately."
    })";

    std::string jsonData = jsonDataStream.str();

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());

        // Set timeouts to avoid freezing issues
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);

        // Configure response handling
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        // Perform request
        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        else {
            std::cout << "AI Response: " << readBuffer << std::endl;
        }

        // Cleanup
        curl_easy_cleanup(curl);
    }
    else {
        std::cerr << "curl_easy_init() failed! Check cURL installation." << std::endl;
    }

    return readBuffer;
}
