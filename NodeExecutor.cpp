#include "NodeExecutor.h"
#include <iostream>
#include <sstream>
#include <windows.h>

NodeExecutor::NodeExecutor(const std::string& nodePath) : nodeExePath(nodePath) {}

bool NodeExecutor::executeScript(const std::string& scriptPath) {
    // Ensure monitorProcess.js is running
    if (!isScriptRunning("monitorProcess.js")) {
        std::cout << "Starting monitorProcess.js..." << std::endl;
        if (!launchProcess("\"" + nodeExePath + "\" \"monitorProcess.js\"")) {
            std::cerr << "Failed to start monitorProcess.js." << std::endl;
            return false;
        }
        Sleep(2000); // Allow initialization
    }
    else {
        std::cout << "monitorProcess.js is already running." << std::endl;
    }

    // Execute get_pid.js
    std::cout << "Executing get_pid.js..." << std::endl;
    if (!launchProcess("\"" + nodeExePath + "\" \"get_pid.js\"")) {
        std::cerr << "Failed to execute get_pid.js." << std::endl;
        return false;
    }

    // Execute Powershell script (launchArmageddon.js)
    std::cout << "Executing launchArmageddon.js..." << std::endl;
    return launchProcess("\"" + nodeExePath + "\" \"launchArmageddon.js\"");
}

bool NodeExecutor::launchProcess(const std::string& command) {
    STARTUPINFOA si = { sizeof(STARTUPINFOA) };
    PROCESS_INFORMATION pi = { 0 };

    if (!CreateProcessA(nullptr, const_cast<char*>(command.c_str()), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        std::cerr << "Failed to execute Node.js script. Error: " << GetLastError() << std::endl;
        return false;
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return true;
}

bool NodeExecutor::isScriptRunning(const std::string& scriptName) {
    std::string command = "tasklist /FI \"IMAGENAME eq node.exe\" /FO CSV | findstr " + scriptName;
    int result = system(command.c_str());
    return (result == 0);
}
