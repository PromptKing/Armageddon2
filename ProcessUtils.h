#ifndef PROCESS_UTILS_H
#define PROCESS_UTILS_H

#include <windows.h>
#include <string>

// Function to find a process by name and get its handle
HANDLE GetProcessHandle(const std::string& processName);

#endif // PROCESS_UTILS_H
