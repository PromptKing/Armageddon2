#ifndef SCREEN_CAPTURE_H
#define SCREEN_CAPTURE_H

#include <vector>

// Function to capture screen using Direct3D 12
bool GetScreenDataD3D12(float* buffer, int width, int height);

// Function to capture screen using Direct3D 11
bool GetScreenDataD3D11(float* buffer, int width, int height);

// Function to capture screen data
// Fills the provided buffer with RGB pixel data
// Returns true on success, false otherwise
bool GetScreenData(float* buffer, int width, int height);

#endif // SCREEN_CAPTURE_H
