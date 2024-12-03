// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>

template <typename dtype>
Tensor conv2d::mps(Tensor input, const Tensor &kernel,
                   const std::optional<const Tensor> &bias,
                   const uint padding_h, const uint padding_w,
                   const uint stride_h, const uint stride_w,
                   const uint dilation_h, const uint dilation_w,
                   const PaddingMode padding_mode, const uint groups) {
    (void)input;
    (void)kernel;
    (void)bias;
    (void)padding_h;
    (void)padding_w;
    (void)stride_h;
    (void)stride_w;
    (void)dilation_h;
    (void)dilation_w;
    (void)padding_mode;
    (void)groups;
    errs::bail("mps support required to use the mps algorithm");
}
template Tensor conv2d::mps<float>(CONV2D_PARAMS);
template Tensor conv2d::mps<double>(CONV2D_PARAMS);
