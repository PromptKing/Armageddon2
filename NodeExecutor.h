#ifndef NODEEXECUTOR_H
#define NODEEXECUTOR_H

#include <string>

class NodeExecutor {
public:
    NodeExecutor(const std::string& nodePath);
    bool executeScript(const std::string& scriptPath);
    bool isScriptRunning(const std::string& scriptName);
private:
    std::string nodeExePath;
    bool launchProcess(const std::string& command);
};

#endif // NODEEXECUTOR_H
