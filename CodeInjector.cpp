#include "CodeInjector.h"
#include <iostream>

// Inject code into the application buffer
bool InjectCode(HANDLE hProcess, InjectionPayload& payload) {
    if (!payload.codeStartAddress || payload.cpuInstructions.empty()) {
        std::cerr << "Error: Invalid payload. Ensure codeStartAddress and CPU instructions are set.\n";
        return false;
    }

    // Allocate memory in the target process for the injected code
    LPVOID remoteMemory = VirtualAllocEx(
        hProcess, nullptr, payload.codeSize, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    if (!remoteMemory) {
        std::cerr << "Error: Failed to allocate memory in target process. Error code: " << GetLastError() << "\n";
        return false;
    }

    // Write the CPU instructions into the allocated memory
    SIZE_T bytesWritten = 0;
    if (!WriteProcessMemory(hProcess, remoteMemory, payload.cpuInstructions.data(), payload.codeSize, &bytesWritten)) {
        std::cerr << "Error: Failed to write CPU instructions to memory. Error code: " << GetLastError() << "\n";
        VirtualFreeEx(hProcess, remoteMemory, 0, MEM_RELEASE);
        return false;
    }

    if (bytesWritten != payload.codeSize) {
        std::cerr << "Warning: Partial write of CPU instructions. Bytes written: " << bytesWritten
            << ", Expected: " << payload.codeSize << "\n";
        VirtualFreeEx(hProcess, remoteMemory, 0, MEM_RELEASE);
        return false;
    }

    // Update the payload to reflect the new memory location
    payload.codeStartAddress = remoteMemory;

    std::cout << "Code injected successfully. Remote memory address: " << remoteMemory << "\n";
    return true;
}

// Create CPU instructions to search CUDA for threads with AVX and x64 support
bool CreateCPUInstructions(InjectionPayload& payload, const std::string& searchPattern) {
    if (searchPattern.empty()) {
        std::cerr << "Error: Search pattern cannot be empty.\n";
        return false;
    }

    // Generate CPU instructions with AVX and x64 architecture compatibility
    payload.cpuInstructions = {
        0x48, 0x89, 0xE5,       // mov rbp, rsp (set up base pointer for stack frame)
        0x48, 0x83, 0xEC, 0x20, // sub rsp, 32 (allocate stack space)

        // Load AVX registers with rendering data
        0xC5, 0xF9, 0x6F, 0xC0, // vmovdqa xmm0, xmmword ptr [rax] (load data into xmm0)
        0xC5, 0xF9, 0x6F, 0xC8, // vmovdqa xmm1, xmmword ptr [rcx] (load data into xmm1)

        // Perform parallel computation for rendering
        0xC5, 0xF9, 0x58, 0xC1, // vaddps xmm0, xmm0, xmm1 (vector add floating-point values)
        0xC5, 0xF9, 0x59, 0xC1, // vmulps xmm0, xmm0, xmm1 (vector multiply floating-point values)

        // Store the result back
        0xC5, 0xF9, 0x7F, 0x00, // vmovdqa [rax], xmm0 (store back the result)

        // End function and return
        0x48, 0x89, 0xEC,       // mov rsp, rbp (restore stack pointer)
        0xC3                    // ret (return from function)
    };

    // Update payload size
    payload.codeSize = payload.cpuInstructions.size();
    std::cout << "Updated CPU instruction payload successfully. Size: " << payload.codeSize << " bytes.\n";
    return true;
}
