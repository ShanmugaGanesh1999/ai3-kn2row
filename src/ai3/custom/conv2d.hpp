#pragma once

#include <ai3.hpp>
#include <vector>
#include <algorithm>

/**
 * @DEFAULT_BOOL{Conv2D}
 */
const bool DEFAULT_CONV2D = true;

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
    // Input tensor dimensions
    auto input_dims = input.shape();
    auto kernel_dims = kernel.shape();
    
    // Validate input and kernel dimensions
    if (input_dims.size() != 4 || kernel_dims.size() != 4) {
        throw std::invalid_argument("Input and kernel must be 4D tensors");
    }
    
    // Extract dimensions
    uint batch_size = input_dims[0];
    uint input_channels = input_dims[1];
    uint input_height = input_dims[2];
    uint input_width = input_dims[3];
    
    uint output_channels = kernel_dims[0];
    uint kernel_height = kernel_dims[2];
    uint kernel_width = kernel_dims[3];
    
    // Handle padding
    uint padded_height = input_height + 2 * padding_h;
    uint padded_width = input_width + 2 * padding_w;
    
    // Calculate output dimensions
    uint output_height = 1 + (padded_height - kernel_height) / stride_h;
    uint output_width = 1 + (padded_width - kernel_width) / stride_w;
    
    // Perform im2row (kn2row) transformation
    // Create a matrix where each row represents a flattened receptive field
    Tensor im2row_matrix(
        {batch_size, output_height * output_width, 
         kernel_height * kernel_width * input_channels / groups}
    );
    
    // Iterate through each batch and channel group
    for (uint b = 0; b < batch_size; ++b) {
        for (uint g = 0; g < groups; ++g) {
            // Group-specific channel ranges
            uint group_input_start = g * (input_channels / groups);
            uint group_input_end = (g + 1) * (input_channels / groups);
            
            // Iterate through output spatial locations
            for (uint h = 0; h < output_height; ++h) {
                for (uint w = 0; w < output_width; ++w) {
                    // Calculate the starting position in the input for this receptive field
                    int start_h = h * stride_h - padding_h;
                    int start_w = w * stride_w - padding_w;
                    
                    // Flattened row for im2row matrix
                    std::vector<dtype> row_data;
                    
                    // Extract receptive field
                    for (uint kh = 0; kh < kernel_height; ++kh) {
                        for (uint kw = 0; kw < kernel_width; ++kw) {
                            for (uint c = group_input_start; c < group_input_end; ++c) {
                                // Calculate actual input coordinates
                                int input_h = start_h + kh * dilation_h;
                                int input_w = start_w + kw * dilation_w;
                                
                                // Handle padding modes
                                dtype pixel_value = 0;
                                bool is_valid_pixel = (input_h >= 0 && input_h < input_height &&
                                                      input_w >= 0 && input_w < input_width);
                                
                                if (is_valid_pixel) {
                                    pixel_value = input[{b, c, input_h, input_w}].item<dtype>();
                                } else {
                                    switch (padding_mode) {
                                        case PaddingMode::ZERO:
                                            pixel_value = 0;
                                            break;
                                        case PaddingMode::REFLECT:
                                            // Reflect padding logic
                                            input_h = reflect_pad(input_h, input_height);
                                            input_w = reflect_pad(input_w, input_width);
                                            pixel_value = input[{b, c, input_h, input_w}].item<dtype>();
                                            break;
                                        case PaddingMode::REPLICATE:
                                            // Clamp padding logic
                                            input_h = std::max(0, std::min(input_h, input_height - 1));
                                            input_w = std::max(0, std::min(input_w, input_width - 1));
                                            pixel_value = input[{b, c, input_h, input_w}].item<dtype>();
                                            break;
                                    }
                                }
                                
                                row_data.push_back(pixel_value);
                            }
                        }
                    }
                    
                    // Set the row in im2row matrix
                    uint row_idx = b * (output_height * output_width) + h * output_width + w;
                    im2row_matrix.slice({row_idx}) = Tensor(row_data);
                }
            }
        }
    }
    
    // Reshape kernel for grouped convolution
    Tensor reshaped_kernel = kernel.reshape({groups, output_channels / groups, 
                                             input_channels / groups, 
                                             kernel_height, kernel_width});
    
    // Perform matrix multiplication
    Tensor output = im2row_matrix.matmul(
        reshaped_kernel.reshape({groups, output_channels / groups, -1}).transpose({1, 2, 0})
    );
    
    // Reshape output to expected dimensions
    output = output.reshape({batch_size, output_channels, output_height, output_width});
    
    // Add bias if provided
    if (bias.has_value()) {
        output = output + bias.value().reshape({1, output_channels, 1, 1});
    }
    
    return output;
}

// Helper function for reflect padding
int reflect_pad(int x, int limit) {
    if (x < 0) x = -x;
    if (x >= limit) x = 2 * limit - x - 2;
    return x;
}