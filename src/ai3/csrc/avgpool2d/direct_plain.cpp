// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>
#include <optional>

template <typename dtype>
Tensor avgpool2d::direct(Tensor input, const uint kernel_h, const uint kernel_w,
                         const uint padding_h, const uint padding_w,
                         const uint stride_h, const uint stride_w,
                         const bool ceil_mode, const bool count_include_pad,
                         const std::optional<int> divisor_override) {
    const uint input_channels = input.input_channels();
    const uint input_height = input.height();
    const uint input_width = input.width();

    const uint output_channels = input_channels;

    const uint output_height = output_hw_for_2d<dtype>(
        input_height, kernel_h, padding_h, std::nullopt, stride_h, ceil_mode);
    const uint output_width = output_hw_for_2d<dtype>(
        input_width, kernel_w, padding_w, std::nullopt, stride_w, ceil_mode);

    Tensor output;
    uint num_samples;
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

    const bool has_divisor_override = divisor_override.has_value();
    const dtype *in_data = data_as<dtype>(input.data);
    dtype *out_data = data_as<dtype>(output.data);
    for (uint samp = 0; samp < num_samples; samp++) {
        for (uint out_c = 0; out_c < output_channels; out_c++) {
            for (uint out_h = 0; out_h < output_height; out_h++) {
                for (uint out_w = 0; out_w < output_width; out_w++) {
                    dtype total = 0;
                    uint pooled_count = 0;
                    for (uint kern_r = 0; kern_r < kernel_h; ++kern_r) {
                        for (uint kern_c = 0; kern_c < kernel_w; ++kern_c) {
                            uint h_offset =
                                out_h * stride_h - padding_h + kern_r;
                            uint w_offset =
                                out_w * stride_w - padding_w + kern_c;

                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                total += in_data[to_linear(
                                    samp, out_c, h_offset, w_offset,
                                    output_channels, input_height,
                                    input_width)];
                                pooled_count++;
                            }
                        }
                    }

                    dtype val;
                    if (has_divisor_override) {
                        val = total / (*divisor_override);
                    } else {
                        if (count_include_pad) {
                            uint hstart = out_h * stride_h - padding_h;
                            uint wstart = out_w * stride_w - padding_w;
                            uint hend = hstart + kernel_h;
                            if (hend > input_height + padding_h) {
                                hend = input_height + padding_h;
                            }
                            uint wend = wstart + kernel_w;
                            if (wend > input_width + padding_w) {
                                wend = input_width + padding_w;
                            }
                            const int pool_size =
                                (hend - hstart) * (wend - wstart);
                            val = total / pool_size;
                        } else {
                            val = total / pooled_count;
                        }
                    }
                    out_data[to_linear(samp, out_c, out_h, out_w,
                                       output_channels, output_height,
                                       output_width)] = val;
                }
            }
        }
    }

    return output;
}

template Tensor avgpool2d::direct<float>(AVGPOOL2D_PARAMS);
template Tensor avgpool2d::direct<double>(AVGPOOL2D_PARAMS);
