#ifndef METADATA_H
#define METADATA_H

#include <string>

struct Metadata {
    int threadID;                  // Unique thread identifier
    double startTime;              // Start timestamp of the operation
    double endTime;                // End timestamp of the operation
    double executionTime;          // Total execution time
    float computationResult;       // Result of the computation
    float predictedResult;         // Predicted result based on prior calculations
    float predictionError;         // Difference between computed and predicted results
    std::string applicationName;   // Name of the application utilizing the resource
    std::string memoryAddress;     // Memory address or resource identifier

    Metadata()
        : threadID(0),
        startTime(0.0),
        endTime(0.0),
        executionTime(0.0),
        computationResult(0.0f),
        predictedResult(0.0f),
        predictionError(0.0f),
        applicationName(""),
        memoryAddress("") {
    }
};

#endif // METADATA_H
