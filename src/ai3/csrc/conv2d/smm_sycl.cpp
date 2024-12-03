// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <ai3.hpp>
#include <algos.hpp>
#include <optional>
using namespace cl;

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
        num_samples = input.batch_size(sample_dims::CONV2D);
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
    sycl::queue *queue_ptr = static_cast<sycl::queue *>(Context::sycl_queue());
    sycl::queue queue = *queue_ptr;

    const bool has_bias = bias.has_value();
    sycl::buffer<dtype> bias_buf =
        has_bias
            ? sycl::buffer<dtype>(data_as<dtype>(bias->data), bias->count())
            : sycl::buffer<dtype>(sycl::range<1>(0));

    sycl::buffer<dtype> buf_cols(num_samples * col_height * col_width);

    sycl::buffer<dtype> buf_input(data_as<dtype>(input.data), input.count());
    sycl::buffer<dtype> buf_kernel(data_as<dtype>(kernel.data), kernel.count());

    sycl::buffer<dtype> buf_output(data_as<dtype>(output.data), output.count());

    const uint max_work_group_size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    const uint kernel_area = kernel_height * kernel_width;

    uint each_kernel = max_work_group_size;
    if (GROUP_SIZE_GUESS < each_kernel) {
        each_kernel = GROUP_SIZE_GUESS;
    }
    if (kernel_area < each_kernel) {
        each_kernel = kernel_area;
    }

    const uint total_kernel =
        ((kernel_area + each_kernel - 1) / each_kernel) * each_kernel;

    queue.submit([&](sycl::handler &h) {
        sycl::accessor acc_input =
            buf_input.template get_access<sycl::access::mode::read>(h);
        sycl::accessor acc_cols =
            buf_cols.template get_access<sycl::access::mode::write>(h);

        h.parallel_for(
            sycl::nd_range(
                sycl::range(num_samples, input_channels, total_kernel),
                sycl::range(1, 1, each_kernel)),
            [=](sycl::nd_item<3> item) {
                uint samp = item.get_global_id(0);
                uint in_c = item.get_global_id(1);
                uint ker = item.get_global_id(2);
                if (ker >= kernel_area)
                    return;
                uint ker_w = ker % kernel_width;
                uint ker_h = ker / kernel_width;

                for (uint out_h = 0; out_h < output_height; ++out_h) {
                    for (uint out_w = 0; out_w < output_width; ++out_w) {
                        uint h_offset =
                            out_h * stride_h - padding_h + ker_h * dilation_h;
                        uint w_offset =
                            out_w * stride_w - padding_w + ker_w * dilation_w;
                        uint col_index = to_linear(
                            samp, out_h, out_w, in_c, ker, output_height,
                            output_width, input_channels, kernel_area);
                        if (h_offset >= 0 && h_offset < input_height &&
                            w_offset >= 0 && w_offset < input_width) {
                            acc_cols[col_index] = acc_input[to_linear(
                                samp, in_c, h_offset, w_offset, input_channels,
                                input_height, input_width)];
                        } else {
                            acc_cols[col_index] = 0;
                        }
                    }
                }
            });
    });
    queue.wait_and_throw();

    const uint output_area = output_height * output_width;
    uint output_work_group_size = max_work_group_size;
    if (GROUP_SIZE_GUESS < output_work_group_size) {
        output_work_group_size = GROUP_SIZE_GUESS;
    }

    uint each_channel, each_output_area, total_output_channels,
        total_output_area;
    proportionate_2d_work_split<dtype>(
        output_channels, output_area, output_work_group_size, &each_channel,
        &each_output_area, &total_output_channels, &total_output_area);

    queue.submit([&](sycl::handler &h) {
        sycl::accessor acc_kernel =
            buf_kernel.template get_access<sycl::access::mode::read>(h);
        sycl::accessor acc_cols =
            buf_cols.template get_access<sycl::access::mode::read>(h);
        sycl::accessor acc_bias =
            bias_buf.template get_access<sycl::access::mode::read>(h);
        sycl::accessor acc_output =
            buf_output.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(
            sycl::nd_range(sycl::range(num_samples, total_output_channels,
                                       total_output_area),
                           sycl::range(1, each_channel, each_output_area)),
            [=](sycl::nd_item<3> item) {
                uint samp = item.get_global_id(0);
                uint out_c = item.get_global_id(1);
                uint out_id = item.get_global_id(2);
                if (out_id >= output_area || out_c >= output_channels)
                    return;
                dtype res = 0;
                const uint col_base =
                    samp * col_width * col_height + out_id * col_height;
                const uint ker_base = out_c * col_height;
                for (uint col_h = 0; col_h < col_height; col_h++) {
                    res += acc_cols[col_base + col_h] *
                           acc_kernel[ker_base + col_h];
                }
                if (has_bias) {
                    res += acc_bias[out_c];
                }
                acc_output[to_linear(samp, out_c, out_id, output_channels,
                                     output_area)] = res;
            });
    });
    queue.wait_and_throw();

    return output;
}

template Tensor conv2d::smm<float>(CONV2D_PARAMS);
template Tensor conv2d::smm<double>(CONV2D_PARAMS);
