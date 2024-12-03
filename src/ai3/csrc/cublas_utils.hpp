// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename T> inline cudaDataType cublas_data_type();

template <> inline cudaDataType cublas_data_type<float>() { return CUDA_R_32F; }

template <> inline cudaDataType cublas_data_type<double>() {
    return CUDA_R_64F;
}

#define CUDA_CHECK(status)                                                     \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(status)              \
                  << std::endl;                                                \
        exit(1);                                                               \
    }

#define CUBLAS_CHECK(status)                                                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "cuBLAS error: " << status << std::endl;                  \
        exit(1);                                                               \
    }
