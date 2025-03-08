#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <chrono>
#include <thread>

HDC hScreenDC;
HDC hMemoryDC;
HBITMAP hBitmap;
HBITMAP hOldBitmap;
BITMAPINFOHEADER bi;

// Capture screen function
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

    // Process with Tesseract here
    processImageWithTesseract(img);

    // Cleanup
    SelectObject(hMemoryDC, hOldBitmap);
    DeleteObject(hBitmap);
    DeleteDC(hMemoryDC);
    ReleaseDC(NULL, hScreenDC);
}

// Process the captured image with Tesseract
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
