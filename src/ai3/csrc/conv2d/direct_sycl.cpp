// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <ai3.hpp>
#include <algos.hpp>
#include <optional>
using namespace cl;

template <typename dtype>
Tensor conv2d::direct(Tensor input, const Tensor &kernel,
                      const std::optional<const Tensor> &bias,
                      const uint padding_h, const uint padding_w,
                      const uint stride_h, const uint stride_w,
                      const uint dilation_h, const uint dilation_w,
                      const PaddingMode padding_mode, uint groups) {
    ensure_same_type(input, kernel, bias);
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
        num_samples = input.batch_size(sample_dims::CONV2D);
        output =
            Tensor({num_samples, output_channels, output_height, output_width},
                   input.scalar_type);
    } else {
        num_samples = 1;
        output = Tensor({output_channels, output_height, output_width},
                        input.scalar_type);
    }

    sycl::queue *queue_ptr = static_cast<sycl::queue *>(Context::sycl_queue());
    sycl::queue queue = *queue_ptr;

    sycl::buffer<dtype> buf_input(data_as<dtype>(input.data), input.count());
    sycl::buffer<dtype> buf_ker(data_as<dtype>(kernel.data), kernel.count());
    sycl::buffer<dtype> buf_output(data_as<dtype>(output.data), output.count());
    const bool has_bias = bias.has_value();
    sycl::buffer<dtype> buf_bias =
        has_bias
            ? sycl::buffer<dtype>(data_as<dtype>(bias->data), bias->count())
            : sycl::buffer<dtype>(sycl::range<1>(0));

    const uint output_area = output_height * output_width;
    const uint max_work_group_size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    uint work_group_size = max_work_group_size;
    if (GROUP_SIZE_GUESS < work_group_size) {
        work_group_size = GROUP_SIZE_GUESS;
    }
    if (output_area < work_group_size) {
        work_group_size = output_area;
    }
    uint each_channel, each_output_area, total_channels, total_output_area;
    proportionate_2d_work_split<dtype>(
        output_channels, output_area, work_group_size, &each_channel,
        &each_output_area, &total_channels, &total_output_area);
    queue.submit([&](sycl::handler &h) {
        auto acc_input =
            buf_input.template get_access<sycl::access::mode::read>(h);
        auto acc_bias =
            buf_bias.template get_access<sycl::access::mode::read>(h);
        auto acc_kernel =
            buf_ker.template get_access<sycl::access::mode::read>(h);
        auto acc_output =
            buf_output.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(
            sycl::nd_range(
                sycl::range(num_samples, total_channels, total_output_area),
                sycl::range(1, each_channel, each_output_area)),
            [=](sycl::nd_item<3> item) {
                const uint samp = item.get_global_id(0);
                const uint out_c = item.get_global_id(1);
                if (out_c >= output_channels) {
                    return;
                }
                const uint area_id = item.get_global_id(2);
                if (area_id >= output_area) {
                    return;
                }
                const uint out_h = area_id / output_width;
                const uint out_w = area_id % output_width;
                dtype sum = 0;
                uint h_offset_base = out_h * stride_h - padding_h;
                uint w_offset_base = out_w * stride_w - padding_w;
                for (uint in_c = 0; in_c < input_channels; ++in_c) {
                    for (uint ker_h = 0; ker_h < kernel_height; ++ker_h) {
                        for (uint ker_w = 0; ker_w < kernel_width; ++ker_w) {
                            uint h_offset = h_offset_base + ker_h * dilation_h;
                            uint w_offset = w_offset_base + ker_w * dilation_w;
                            if (h_offset >= 0 && h_offset < input_height &&
                                w_offset >= 0 && w_offset < input_width) {
                                sum += acc_input[to_linear(
                                           samp, in_c, h_offset, w_offset,
                                           input_channels, input_height,
                                           input_width)] *
                                       acc_kernel[to_linear(
                                           out_c, in_c, ker_h, ker_w,
                                           input_channels, kernel_height,
                                           kernel_width)];
                            }
                        }
                    }
                }
                if (has_bias) {
                    sum += acc_bias[out_c];
                }
                acc_output[to_linear(samp, out_c, out_h, out_w, output_channels,
                                     output_height, output_width)] = sum;
            });
    });

    queue.wait_and_throw();

    return output;
}

template Tensor conv2d::direct<float>(CONV2D_PARAMS);
template Tensor conv2d::direct<double>(CONV2D_PARAMS);
