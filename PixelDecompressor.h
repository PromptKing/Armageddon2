#ifndef PIXEL_DECOMPRESSOR_H
#define PIXEL_DECOMPRESSOR_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <windows.h>
#include <iostream>

class PixelDecompressor {
public:
    PixelDecompressor(int width, int height);
    ~PixelDecompressor();

    void captureScreen();
    void decompressPixels();
    void enhanceSharpness();
    void display();

private:
    int width, height;
    cv::Mat screenImage;
    cv::Mat decompressedImage;
};

#endif // PIXEL_DECOMPRESSOR_H
