#ifndef UTILITY_H
#define UTILITY_H

#include <windows.h>
#include <vector>

#include <cuda_runtime.h>


// Declare ModifyMemory
bool ModifyMemory(HANDLE hProcess, LPVOID targetBaseAddress, const std::vector<unsigned char>& newData);

// Declare ValidateMemoryAccess
bool ValidateMemoryAccess(HANDLE hProcess, LPVOID address);

#endif // UTILITY_H
