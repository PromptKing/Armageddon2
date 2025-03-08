#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <chrono>
#include <thread>
#include <string>
#include <filesystem>

HDC hScreenDC;
HDC hMemoryDC;
HBITMAP hBitmap;
HBITMAP hOldBitmap;
BITMAPINFOHEADER bi;

// Function to capture the screen
void captureScreen() {
    hScreenDC = GetDC(NULL);
    hMemoryDC = CreateCompatibleDC(hScreenDC);

    // Set the width and height of the screen capture area
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    hBitmap = CreateCompatibleBitmap(hScreenDC, screenWidth, screenHeight);
    hOldBitmap = (HBITMAP)SelectObject(hMemoryDC, hBitmap);

    // Capture the screen content into the bitmap
    BitBlt(hMemoryDC, 0, 0, screenWidth, screenHeight, hScreenDC, 0, 0, SRCCOPY);

    // Convert the captured image into a Mat (OpenCV format)
    cv::Mat img(screenHeight, screenWidth, CV_8UC4); // Assuming 32-bit (RGBA) color depth
    GetDIBits(hMemoryDC, hBitmap, 0, screenHeight, img.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    // Process the image with Tesseract
    processImageWithTesseract(img);

    // Save the captured image to the specified directory
    saveImage(img);

    // Cleanup
    SelectObject(hMemoryDC, hOldBitmap);
    DeleteObject(hBitmap);
    DeleteDC(hMemoryDC);
    ReleaseDC(NULL, hScreenDC);
}

// Function to save the image as a JPEG file
void saveImage(const cv::Mat& img) {
    // Define the directory path
    std::string directory = "C:\\Windows\\System32\\Armageddon2 DLLs\\Tesseract Real Time Screen Capture";

    // Ensure the directory exists
    if (!std::filesystem::exists(directory)) {
        std::filesystem::create_directory(directory);
    }

    // Get the current timestamp for unique filenames
    auto timestamp = std::chrono::system_clock::now();
    std::time_t now = std::chrono::system_clock::to_time_t(timestamp);
    char filename[100];
    std::strftime(filename, sizeof(filename), "%Y%m%d%H%M%S.jpg", std::localtime(&now));

    // Create the full file path
    std::string filePath = directory + "\\" + filename;

    // Save the image as JPEG
    if (cv::imwrite(filePath, img)) {
        std::cout << "Saved screen capture as: " << filePath << std::endl;
    }
    else {
        std::cerr << "Failed to save image.\n";
    }
}

// Function to process the image with Tesseract
void processImageWithTesseract(const cv::Mat& img) {
    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng"); // Initialize tesseract with English language

    cv::Mat grayscale;
    cv::cvtColor(img, grayscale, cv::COLOR_BGRA2GRAY);  // Convert to grayscale

    tess.SetImage(grayscale.data, grayscale.cols, grayscale.rows, 1, grayscale.cols);
    char* text = tess.GetUTF8Text();

    std::cout << "Recognized Text: " << text << std::endl;

    delete[] text; // Free memory
}

int main() {
    while (true) {
        captureScreen();  // Capture the screen every frame

        // Sleep for a bit before capturing the next screen (control the FPS)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}
