#pragma once
#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include "ObjectStructures.h"
#include <vector>
#include <string>

// OpenCV Support
#ifdef HAVE_OPENCV_OBJDETECT
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#endif

// Structure for a 3D vertex
struct Vertex3D {
    float x, y, z;
};

// Structure for an object’s bounding box
struct BoundingBox {
    Vertex3D min; // Minimum corner
    Vertex3D max; // Maximum corner
};

// Class for 3D object detection
class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector();

    // Set the 3D scene data
    void SetSceneData(const std::vector<BoundingBox>& objects);

    // Detect objects in the scene
    std::vector<std::string> DetectObjects(const Vertex3D& queryPoint, float detectionRadius);

    // Check for collisions between objects
    bool CheckCollision(const BoundingBox& box1, const BoundingBox& box2);

private:
    std::vector<BoundingBox> sceneObjects; // List of objects in the scene

#ifdef HAVE_OPENCV_OBJDETECT
    cv::Mat detectionMap; // OpenCV-based detection map for 2D object detection
#endif
};

#endif // OBJECTDETECTOR_H
