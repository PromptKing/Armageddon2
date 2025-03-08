#ifndef HOSTVERIFIER_H
#define HOSTVERIFIER_H

#include <string>
#include <vector>

struct CPUHostInfo {
    std::string cpuName;
    int coreCount;
    int threadCount;
    bool avxSupport;
    bool avx2Support;
    bool avx512Support;
    std::vector<std::string> additionalFeatures;
};


// Host information structure
struct HostInfo {
    std::string cpuName;
    int coreCount;
    int threadCount;
    bool avxSupport;
    bool avx2Support;
    bool avx512Support;
    std::vector<std::string> additionalFeatures;
};

// Class to verify and retrieve host CPU information
class HostVerifier {
public:
    HostVerifier();
    bool VerifyHostCompatibility() const;
    HostInfo GetHostInfo() const;

private:
    HostInfo hostInfo;
    void RetrieveHostInfo();
    bool CheckFeatureSupport(const std::string& feature) const;
};

#endif // HOSTVERIFIER_H
