// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>
#include <optional>

template <typename dtype>
inline Tensor linear::gemm(Tensor input, const Tensor &weight,
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

    const bool has_bias = bias.has_value();
    const dtype *bias_data = nullptr;
    if (has_bias) {
        bias_data = data_as<dtype>(bias->data);
    }
    const dtype *in_data = data_as<dtype>(input.data);
    const dtype *weight_data = data_as<dtype>(weight.data);
    dtype *out_data = data_as<dtype>(output.data);
    for (uint s = 0; s < num_samples; s++) {
        for (uint i = 0; i < out_features; i++) {
            dtype res = 0;
            for (uint j = 0; j < in_features; ++j) {
                res += weight_data[to_linear(i, j, in_features)] *
                       in_data[to_linear(s, j, in_features)];
            }
            if (has_bias) {
                res += bias_data[i];
            }
            out_data[to_linear(s, i, out_features)] = res;
        }
    }
    return output;
}

template Tensor linear::gemm<float>(LINEAR_PARAMS);
template Tensor linear::gemm<double>(LINEAR_PARAMS);
