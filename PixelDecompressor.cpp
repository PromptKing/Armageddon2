#include "PixelDecompressor.h"

// Constructor
PixelDecompressor::PixelDecompressor(int width, int height) : width(width), height(height) {
    screenImage = cv::Mat(height, width, CV_8UC4);
    decompressedImage = cv::Mat(height * 2, width * 2, CV_8UC3);
}

// Destructor
PixelDecompressor::~PixelDecompressor() {}

// Capture screen pixels using Windows API
void PixelDecompressor::captureScreen() {
    HDC hdcScreen = GetDC(NULL);
    HDC hdcMemory = CreateCompatibleDC(hdcScreen);
    HBITMAP hBitmap = CreateCompatibleBitmap(hdcScreen, width, height);
    SelectObject(hdcMemory, hBitmap);

    BitBlt(hdcMemory, 0, 0, width, height, hdcScreen, 0, 0, SRCCOPY);

    BITMAPINFOHEADER bi = { sizeof(BITMAPINFOHEADER), width, -height, 1, 32, BI_RGB };
    GetDIBits(hdcMemory, hBitmap, 0, height, screenImage.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    DeleteObject(hBitmap);
    DeleteDC(hdcMemory);
    ReleaseDC(NULL, hdcScreen);
}

// Perform pixel decompression using OpenCV (CPU-based upscaling)
void PixelDecompressor::decompressPixels() {
    cv::resize(screenImage, decompressedImage, cv::Size(width * 2, height * 2), 0, 0, cv::INTER_CUBIC);
}

// Apply sharpening filter to enhance clarity
void PixelDecompressor::enhanceSharpness() {
    cv::Mat sharpened;
    cv::GaussianBlur(decompressedImage, sharpened, cv::Size(0, 0), 5);
    cv::addWeighted(decompressedImage, 2.5, sharpened, -0.3, 0, decompressedImage);
}

// Display the enhanced image
void PixelDecompressor::display() {
    cv::imshow("Decompressed & Enhanced Screen", decompressedImage);
    cv::waitKey(1);
}
