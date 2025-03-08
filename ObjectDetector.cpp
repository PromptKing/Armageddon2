#include "ObjectDetector.h"
#include <cmath>
#include <iostream>

#ifdef HAVE_OPENCV_OBJDETECT
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#endif

// Constructor
ObjectDetector::ObjectDetector() {}

// Destructor
ObjectDetector::~ObjectDetector() {}

// Set the scene data
void ObjectDetector::SetSceneData(const std::vector<BoundingBox>& objects) {
    sceneObjects = objects;
}

// Detect objects within a given radius from the query point
std::vector<std::string> ObjectDetector::DetectObjects(const Vertex3D& queryPoint, float detectionRadius) {
    std::vector<std::string> detectedObjects;

#ifdef HAVE_OPENCV_OBJDETECT
    std::cout << "[INFO] Using OpenCV object detection.\n";

    cv::Mat detectionMap = cv::Mat::zeros(100, 100, CV_8UC1); // Example detection map
    for (size_t i = 0; i < sceneObjects.size(); ++i) {
        const auto& box = sceneObjects[i];

        // Convert 3D bounding box to a 2D detection map
        int x = static_cast<int>((box.min.x + box.max.x) / 2);
        int y = static_cast<int>((box.min.y + box.max.y) / 2);

        if (x >= 0 && x < detectionMap.cols && y >= 0 && y < detectionMap.rows) {
            detectionMap.at<uchar>(y, x) = 255;
            detectedObjects.push_back("Object " + std::to_string(i));
        }
    }
#else
    std::cout << "[INFO] Using standard object detection.\n";

    for (size_t i = 0; i < sceneObjects.size(); ++i) {
        const auto& box = sceneObjects[i];

        // Check if the query point is within the detection radius of the bounding box
        float dx = std::max(box.min.x - queryPoint.x, 0.0f) + std::max(queryPoint.x - box.max.x, 0.0f);
        float dy = std::max(box.min.y - queryPoint.y, 0.0f) + std::max(queryPoint.y - box.max.y, 0.0f);
        float dz = std::max(box.min.z - queryPoint.z, 0.0f) + std::max(queryPoint.z - box.max.z, 0.0f);

        if (std::sqrt(dx * dx + dy * dy + dz * dz) <= detectionRadius) {
            detectedObjects.push_back("Object " + std::to_string(i));
        }
    }
#endif

    return detectedObjects;
}

// Check if two bounding boxes collide
bool ObjectDetector::CheckCollision(const BoundingBox& box1, const BoundingBox& box2) {
    return (box1.min.x <= box2.max.x && box1.max.x >= box2.min.x) &&
        (box1.min.y <= box2.max.y && box1.max.y >= box2.min.y) &&
        (box1.min.z <= box2.max.z && box1.max.z >= box2.min.z);
}
