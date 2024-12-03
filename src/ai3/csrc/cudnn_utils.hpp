// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cudnn.h>
#include <iostream>

template <typename T> inline cudnnDataType_t cudnn_data_type();

template <> inline cudnnDataType_t cudnn_data_type<float>() {
    return CUDNN_DATA_FLOAT;
}

template <> inline cudnnDataType_t cudnn_data_type<double>() {
    return CUDNN_DATA_DOUBLE;
}

#define CUDNN_CHECK(status)                                                    \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status)            \
                  << std::endl;                                                \
        exit(1);                                                               \
    }
