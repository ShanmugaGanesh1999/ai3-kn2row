#pragma once

#include <ai3.hpp>
#include <vector>
#include <optional>
#include <stdexcept>

/**
 * @DEFAULT_BOOL{Conv2D}
 */
const bool DEFAULT_CONV2D = false;

/**
 * @CUSTOM_OP{Conv2D,conv2d}
 */
template <typename dtype>
Tensor conv2d_custom(Tensor input, const Tensor &kernel,
                     const std::optional<const Tensor> &bias,
                     const uint padding_h, const uint padding_w,
                     const uint stride_h, const uint stride_w,
                     const uint dilation_h, const uint dilation_w,
                     const PaddingMode padding_mode, uint groups) {
    // Ensure input types are compatible
    ensure_same_type(input, kernel, bias);
    errs::bail_if(padding_mode != PaddingMode::Zeros, "Padding mode must be zeroes");
    errs::bail_if(groups != 1, "Groups must be 1");

    // Extract dimensions
    const uint input_channels = input.input_channels();
    const uint input_height = input.height();
    const uint input_width = input.width();
    const uint kernel_height = kernel.height();
    const uint kernel_width = kernel.width();
    const uint output_channels = kernel.output_channels();
    const uint output_height = output_hw_for_2d<dtype>(
        input_height, kernel_height, padding_h, dilation_h, stride_h, false);
    const uint output_width = output_hw_for_2d<dtype>(
        input_width, kernel_width, padding_w, dilation_w, stride_w, false);

    // Create output tensor
    Tensor output = Tensor({output_channels, output_height, output_width}, input.scalar_type);

    // Flatten kernel once for reuse
    std::vector<std::vector<dtype>> flattened_kernels(output_channels,
        std::vector<dtype>(input_channels * kernel_height * kernel_width, 0));

    for (uint oc = 0; oc < output_channels; ++oc) {
        for (uint ic = 0; ic < input_channels; ++ic) {
            for (uint kh = 0; kh < kernel_height; ++kh) {
                for (uint kw = 0; kw < kernel_width; ++kw) {
                    flattened_kernels[oc][ic * kernel_height * kernel_width + kh * kernel_width + kw] =
                        static_cast<dtype*>(kernel.data)[oc * input_channels * kernel_height * kernel_width + ic * kernel_height * kernel_width + kh * kernel_width + kw];
                }
            }
        }
    }

    // Process each output pixel
    for (uint oc = 0; oc < output_channels; ++oc) {
        for (uint oh = 0; oh < output_height; ++oh) {
            for (uint ow = 0; ow < output_width; ++ow) {
                dtype value = 0;
                for (uint ic = 0; ic < input_channels; ++ic) {
                    for (uint kh = 0; kh < kernel_height; ++kh) {
                        for (uint kw = 0; kw < kernel_width; ++kw) {
                            int ih = oh * stride_h - padding_h + kh * dilation_h;
                            int iw = ow * stride_w - padding_w + kw * dilation_w;

                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                value += static_cast<dtype*>(input.data)[ic * input_height * input_width + ih * input_width + iw] *
                                         flattened_kernels[oc][ic * kernel_height * kernel_width + kh * kernel_width + kw];
                            }
                        }
                    }
                }
                if (bias.has_value()) {
                    value += static_cast<dtype*>(bias->data)[oc];
                }
                static_cast<dtype*>(output.data)[oc * output_height * output_width + oh * output_width + ow] = value;
            }
        }
    }
    return output;
}

