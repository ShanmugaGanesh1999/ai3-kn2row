// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <ai3.hpp>
#include <algos.hpp>
#include <optional>
using namespace cl;

template <typename dtype>
Tensor linear::gemm(Tensor input, const Tensor &weight,
                    const std::optional<const Tensor> &bias) {
    ensure_same_type(input, weight);
    errs::bail_if(input.width() != weight.width(),
                  "Invalid matrix multiplication: input width=", input.width(),
                  " weight width=", weight.width());
    const uint in_features = input.width();
    const uint out_features = weight.height();

    Tensor output;
    uint num_samples;
    if (input.batched(sample_dims::LINEAR)) {
        num_samples = input.batch_size(sample_dims::LINEAR);
        output = Tensor({num_samples, out_features}, input.scalar_type);
    } else {
        num_samples = 1;
        output = Tensor({out_features}, input.scalar_type);
    }

    sycl::queue *queue_ptr = static_cast<sycl::queue *>(Context::sycl_queue());
    sycl::queue queue = *queue_ptr;

    sycl::buffer<dtype> buf_input(data_as<dtype>(input.data), input.count());
    sycl::buffer<dtype> buf_weight(data_as<dtype>(weight.data), weight.count());
    sycl::buffer<dtype> buf_output(data_as<dtype>(output.data), output.count());

    const bool has_bias = bias.has_value();
    sycl::buffer<dtype> buf_bias =
        has_bias
            ? sycl::buffer<dtype>(data_as<dtype>(bias->data), bias->count())
            : sycl::buffer<dtype>(sycl::range<1>(0));

    const uint max_block_size =
        queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    uint block_size_feature = out_features;
    block_size_feature <<= prev_power_of_two(std::sqrt(max_block_size));
    uint block_size_sample = num_samples;
    block_size_sample <<= prev_power_of_two(std::sqrt(max_block_size));

    uint each_sample, each_feature, total_samples, total_features;
    proportionate_2d_work_split<dtype>(
        num_samples, out_features, prev_power_of_two(std::sqrt(max_block_size)),
        &each_sample, &each_feature, &total_samples, &total_features);

    queue.submit([&](sycl::handler &h) {
        auto acc_input =
            buf_input.template get_access<sycl::access::mode::read>(h);
        auto acc_bias =
            buf_bias.template get_access<sycl::access::mode::read>(h);
        auto acc_weight =
            buf_weight.template get_access<sycl::access::mode::read>(h);
        auto acc_output =
            buf_output.template get_access<sycl::access::mode::write>(h);
        h.parallel_for(
            sycl::nd_range(sycl::range<2>(total_samples, total_features),
                           sycl::range<2>(each_sample, each_feature)),
            [=](sycl::nd_item<2> item) {
                const uint sample_idx = item.get_global_id(0);
                const uint out_idx = item.get_global_id(1);
                if (sample_idx >= num_samples || out_idx >= out_features) {
                    return;
                }

                dtype output_value = 0;

                for (uint in_idx = 0; in_idx < in_features; ++in_idx) {
                    output_value +=
                        acc_input[to_linear(sample_idx, in_idx, in_features)] *
                        acc_weight[to_linear(out_idx, in_idx, in_features)];
                }

                if (has_bias) {
                    output_value += acc_bias[out_idx];
                }

                acc_output[to_linear(sample_idx, out_idx, out_features)] =
                    output_value;
            });
    });

    queue.wait_and_throw();

    return output;
}

template Tensor linear::gemm<float>(LINEAR_PARAMS);
template Tensor linear::gemm<double>(LINEAR_PARAMS);
