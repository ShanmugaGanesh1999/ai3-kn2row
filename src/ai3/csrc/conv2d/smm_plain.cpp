// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>
#include <optional>

template <typename dtype>
Tensor conv2d::smm(Tensor input, const Tensor &kernel,
                   const std::optional<const Tensor> &bias,
                   const uint padding_h, const uint padding_w,
                   const uint stride_h, const uint stride_w,
                   const uint dilation_h, const uint dilation_w,
                   const PaddingMode padding_mode, uint groups) {
    errs::bail_if(padding_mode != PaddingMode::Zeros,
                  "padding mode must be zeroes");
    errs::bail_if(groups != 1, "groups must be 1");

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

    uint num_samples;
    Tensor output;
    if (input.batched(sample_dims::CONV2D)) {
        num_samples = input.batch_size(sample_dims::POOL2D);
        output =
            Tensor({num_samples, output_channels, output_height, output_width},
                   input.scalar_type);
    } else {
        num_samples = 1;
        output = Tensor({output_channels, output_height, output_width},
                        input.scalar_type);
    }

    uint col_height = input_channels * kernel_height * kernel_width;
    uint col_width = output_height * output_width;
    Tensor col({col_height, col_width}, input.scalar_type);

    const bool has_bias = bias.has_value();

    dtype *col_data = data_as<dtype>(col.data);
    const dtype *in_data = data_as<dtype>(input.data);
    const dtype *kern_data = data_as<dtype>(kernel.data);
    dtype *out_data = data_as<dtype>(output.data);
    dtype *bias_data = nullptr;
    if (has_bias) {
        bias_data = data_as<dtype>(bias->data);
    }

    for (uint samp = 0; samp < num_samples; samp++) {
        for (uint in_c = 0; in_c < input_channels; in_c++) {
            for (uint ker_h = 0; ker_h < kernel_height; ker_h++) {
                for (uint ker_w = 0; ker_w < kernel_width; ker_w++) {
                    for (uint out_h = 0; out_h < output_height; ++out_h) {
                        for (uint out_w = 0; out_w < output_width; ++out_w) {
                            uint h_offset = out_h * stride_h - padding_h +
                                            ker_h * dilation_h;
                            uint w_offset = out_w * stride_w - padding_w +
                                            ker_w * dilation_w;
                            uint col_index = to_linear(
                                in_c, ker_h, ker_w, out_h, out_w, kernel_height,
                                kernel_width, output_height, output_width);
                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                col_data[col_index] = in_data[to_linear(
                                    samp, in_c, h_offset, w_offset,
                                    input_channels, input_height, input_width)];
                            } else {
                                col_data[col_index] = 0;
                            }
                        }
                    }
                }
            }
        }

        for (uint out_c = 0; out_c < output_channels; out_c++) {
            for (uint out_h = 0; out_h < output_height; out_h++) {
                for (uint out_w = 0; out_w < output_width; out_w++) {
                    dtype res = 0;
                    for (uint in_c = 0; in_c < input_channels; in_c++) {
                        for (uint ker_h = 0; ker_h < kernel_height; ker_h++) {
                            for (uint ker_w = 0; ker_w < kernel_width;
                                 ker_w++) {
                                res += col_data[to_linear(
                                           in_c, ker_h, ker_w, out_h, out_w,
                                           kernel_height, kernel_width,
                                           output_height, output_width)] *
                                       kern_data[to_linear(
                                           out_c, in_c, ker_h, ker_w,
                                           input_channels, kernel_height,
                                           kernel_width)];
                            }
                        }
                    }
                    if (has_bias) {
                        res += bias_data[out_c];
                    }
                    out_data[to_linear(samp, out_c, out_h, out_w,
                                       output_channels, output_height,
                                       output_width)] = res;
                }
            }
        }
    }

    return output;
}

template Tensor conv2d::smm<float>(CONV2D_PARAMS);
template Tensor conv2d::smm<double>(CONV2D_PARAMS);
