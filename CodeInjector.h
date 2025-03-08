#ifndef CODE_INJECTOR_H
#define CODE_INJECTOR_H

#include <windows.h>
#include <vector>
#include <string>

struct HostInjectionPayload {
    LPVOID codeStartAddress;
    SIZE_T codeSize;
    std::vector<unsigned char> cpuInstructions;
};

struct InjectionPayload {
    LPVOID codeStartAddress; // Starting address of the code to be injected
    SIZE_T codeSize;         // Size of the code to be injected
    std::vector<unsigned char> cpuInstructions; // CPU instructions to search CUDA threads
};

// Declare functions for code injection
bool InjectCode(HANDLE hProcess, InjectionPayload& payload);
bool CreateCPUInstructions(InjectionPayload& payload, const std::string& searchPattern);

#endif // CODE_INJECTOR_H
