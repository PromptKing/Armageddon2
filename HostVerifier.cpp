#include "HostVerifier.h"
#include <iostream>
#include <thread>
#include <intrin.h> // For CPUID

HostVerifier::HostVerifier() {
    RetrieveHostInfo();
}

void HostVerifier::RetrieveHostInfo() {
    // Initialize host info structure
    hostInfo.coreCount = std::thread::hardware_concurrency() / 2; // Assuming hyper-threading
    hostInfo.threadCount = std::thread::hardware_concurrency();

    // Retrieve CPU name
    int cpuInfo[4] = { 0 };
    char cpuBrand[0x40] = { 0 };
    __cpuid(cpuInfo, 0x80000002);
    memcpy(cpuBrand, cpuInfo, sizeof(cpuInfo));
    __cpuid(cpuInfo, 0x80000003);
    memcpy(cpuBrand + 16, cpuInfo, sizeof(cpuInfo));
    __cpuid(cpuInfo, 0x80000004);
    memcpy(cpuBrand + 32, cpuInfo, sizeof(cpuInfo));
    hostInfo.cpuName = cpuBrand;

    // Check for AVX, AVX2, and AVX512 support
    __cpuid(cpuInfo, 1);
    hostInfo.avxSupport = cpuInfo[2] & (1 << 28); // Check AVX bit
    hostInfo.avx2Support = cpuInfo[2] & (1 << 5); // Check AVX2 bit
    hostInfo.avx512Support = false;               // Placeholder for AVX512 detection

    // Add additional features (placeholder logic)
    hostInfo.additionalFeatures.push_back("SSE");
    hostInfo.additionalFeatures.push_back("SSE2");
    if (hostInfo.avxSupport) hostInfo.additionalFeatures.push_back("AVX");
    if (hostInfo.avx2Support) hostInfo.additionalFeatures.push_back("AVX2");
    if (hostInfo.avx512Support) hostInfo.additionalFeatures.push_back("AVX512");
}

bool HostVerifier::CheckFeatureSupport(const std::string& feature) const
{
    return false;
}

bool HostVerifier::VerifyHostCompatibility() const {
    if (!hostInfo.avxSupport) {
        std::cerr << "Error: Host does not support AVX. Cannot proceed.\n";
        return false;
    }
    std::cout << "Host verification successful. CPU is compatible.\n";
    return true;
}

HostInfo HostVerifier::GetHostInfo() const {
    return hostInfo;
}
