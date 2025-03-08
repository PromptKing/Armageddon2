#ifndef AI_INTEGRATION_H
#define AI_INTEGRATION_H

#include <string>

std::string QueryOllamaAI(int threadID, float computationResult, float predictedResult, float lookaheadPrediction, const std::string& modelPath);

#endif // AI_INTEGRATION_H
