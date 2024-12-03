// SPDX-License-Identifier: Apache-2.0

#include <ai3.hpp>
#include <algos.hpp>

template <typename dtype>
Tensor flatten::direct(Tensor input, const uint start_dim, int end_dim_orig) {
    errs::bail_if(end_dim_orig != -1 && int(start_dim) > end_dim_orig,
                  "start dimension > end dimension in flattening function");
    uint end_dim;
    if (end_dim_orig == -1) {
        end_dim = input.shape.size() - 1;
    } else {
        end_dim = end_dim_orig;
    }
    const int new_num_dim = input.shape.size() - (end_dim - start_dim);
    errs::bail_if(
        new_num_dim < 0,
        "tensor would have a negative number of dimensions after flattening");

    std::vector<uint> new_shape(new_num_dim);
    int flat = 1;
    int shift = 0;
    for (uint dim = 0; dim < input.shape.size(); dim++) {
        if (dim < start_dim || dim > end_dim) {
            new_shape[dim - shift] = input.shape[dim];
        } else {
            flat *= input.shape[dim];
            if (dim == end_dim) {
                new_shape[start_dim] = flat;
                shift = end_dim - start_dim;
            }
        }
    }
    input.shape = new_shape;
    return input;
}

template Tensor flatten::direct<float>(FLATTEN_PARAMS);
template Tensor flatten::direct<double>(FLATTEN_PARAMS);
