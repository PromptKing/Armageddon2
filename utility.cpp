#include "utility.h"
#include <cuda_runtime.h>
#include <iostream>
#include <windows.h>


// Modify Memory
bool ModifyMemory(HANDLE hProcess, LPVOID targetBaseAddress, const std::vector<unsigned char>& newData) {
    SIZE_T bytesWritten = 0;

    if (!ValidateMemoryAccess(hProcess, targetBaseAddress)) {
        return false;
    }

    if (!WriteProcessMemory(hProcess, targetBaseAddress, newData.data(), newData.size(), &bytesWritten)) {
        std::cerr << "Error: Failed to write memory. Address: " << targetBaseAddress
            << ", Error Code: " << GetLastError() << std::endl;
        return false;
    }

    if (bytesWritten != newData.size()) {
        std::cerr << "Warning: Incomplete memory write. Bytes Written: " << bytesWritten
            << ", Expected: " << newData.size() << std::endl;
        return false;
    }

    std::cout << "Memory modified successfully. Address: " << targetBaseAddress
        << ", Bytes Written: " << bytesWritten << std::endl;
    return true;
}

bool ValidateMemoryAccess(HANDLE hProcess, LPVOID address)
{
    return false;
}
