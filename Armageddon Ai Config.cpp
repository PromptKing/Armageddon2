#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <vector>
#include <sstream>
#include <string>
#include <cstdlib>
#include <nlohmann/json.hpp> // JSON processing
#include <windows.h>
#include <curl/curl.h>
#include "ArmageddonAlgorithm.h"

using json = nlohmann::json;

// Paths for configuration files and dataset
const std::string INI_FILE = "C:\\Users\\KrunkCiti\\Desktop\\Armageddon2\\Amageddon2\\x64\\Debug\\armageddon_config.ini";
const std::string JSON_FILE = "C:\\Users\\KrunkCiti\\Desktop\\Armageddon2\\Amageddon2\\x64\\Debug\\config.json";
const std::string PYTHON_SCRIPT = "C:\\Users\\KrunkCiti\\AppData\\Local\\Programs\\Python\\Python312\\KodeByteStudio\\MemoryExtractionClaude.py";
const std::string CSV_FILE = "C:\\Users\\KrunkCiti\\AppData\\Local\\Programs\\Python\\Python312\\KodeByteStudio\\MACHINE_CODE_DATASET_CLAUD.csv";

std::mutex ai_mutex;

// Function to execute the Python script
void ExecutePythonScript() {
    std::string command = "python \"" + PYTHON_SCRIPT + "\"";
    std::cout << "[INFO] Running Python script: " << command << std::endl;
    int result = std::system(command.c_str());
    if (result != 0) {
        std::cerr << "[ERROR] Failed to execute Python script!\n";
    }
}

// Function to read the CSV dataset
std::vector<std::string> ReadCSVFile() {
    std::ifstream file(CSV_FILE);
    std::vector<std::string> dataset;
    std::string line;

    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open CSV file: " << CSV_FILE << std::endl;
        return dataset;
    }

    while (std::getline(file, line)) {
        dataset.push_back(line);
    }
    file.close();
    return dataset;
}


// Function to load INI configurations
void LoadINIConfig() {
    std::ifstream file(INI_FILE);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open INI file: " << INI_FILE << std::endl;
        return;
    }

    std::cout << "[INFO] Loaded INI Configuration:\n";
    while (std::getline(file, line)) {
        std::cout << line << std::endl;
    }
    file.close();
}

// Function to load JSON configurations
void LoadJSONConfig() {
    std::ifstream file(JSON_FILE);
    json config;
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open JSON file: " << JSON_FILE << std::endl;
        return;
    }
    file >> config;
    file.close();

    std::cout << "[INFO] Loaded JSON Configuration:\n" << config.dump(4) << std::endl;
}

