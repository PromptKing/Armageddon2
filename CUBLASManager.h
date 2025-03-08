#ifndef CUBLAS_MANAGER_H
#define CUBLAS_MANAGER_H

#include <cublas_v2.h>

struct HostCUBLASHandle {
    cublasHandle_t handle;

    void Initialize() {
        cublasCreate(&handle);
    }

    void Destroy() {
        cublasDestroy(handle);
    }
};

#endif // CUBLAS_MANAGER_H
