// SPDX-License-Identifier: Apache-2.0

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <ai3.hpp>
#include <algos.hpp>

uint count = 0;
double total = 0;

const uint MPSTENSOR_RANK = 4;

inline uint computeMPSAlignOffset(const uint kernel, const uint padding,
                                  const uint dilation = 1) {
    return (((kernel - 1) * dilation + 1) / 2) - padding;
}

void *gen_mps_graph_device(void) {
    MPSGraphDevice *device =
        [MPSGraphDevice deviceWithMTLDevice:MTLCreateSystemDefaultDevice()];
    return device;
}

void *gen_mps_graph(void) {
    MPSGraph *graph = [MPSGraph new];
    return graph;
}

void release_mps_graph(void *g) {
    MPSGraph *graph = (MPSGraph *)g;
    [graph release];
}

MPSShape *mps_shape(const std::vector<uint> &shape, const bool bias) {
    NSMutableArray *ret =
        [[[NSMutableArray alloc] initWithCapacity:MPSTENSOR_RANK] autorelease];
    uint i = 0;
    uint shift = 0;
    if (bias) {
        [ret addObject:@1];
        [ret addObject:@(shape[0])];
        [ret addObject:@1];
        [ret addObject:@1];
    } else {
        if (shape.size() < MPSTENSOR_RANK) {
            ret[0] = @1;
            i = 1;
            shift = 1;
        }
        for (; i < shape.size() + shift; i++) {
            ret[i] = @(shape[i - shift]);
        }
    }

    return ret;
}

struct MPSTensor {
    MPSGraphTensor *placeholder;
    MPSGraphTensorData *data;
};

MPSGraphTensorData *output_tensor_data(MPSGraphDevice *device,
                                       MPSGraphTensor *placeholder,
                                       const Tensor &tens) {
    id<MTLBuffer> output_buffer = [[device metalDevice]
        newBufferWithBytesNoCopy:tens.data
                          length:sizeof(float) * tens.count()
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    MPSGraphTensorData *output_data = [[[MPSGraphTensorData alloc]
        initWithMTLBuffer:output_buffer
                    shape:[placeholder shape]
                 dataType:MPSDataTypeFloat32] autorelease];
    return output_data;
}

MPSTensor feed_tensor(MPSGraph *graph, MPSGraphDevice *device,
                      const Tensor &tens, const bool bias = false) {
    MPSGraphTensor *placeholder =
        [graph placeholderWithShape:mps_shape(tens.shape, bias)
                           dataType:MPSDataTypeFloat32
                               name:nil];
    id<MTLBuffer> buffer = [[device metalDevice]
        newBufferWithBytesNoCopy:tens.data
                          length:sizeof(float) * tens.count()
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    MPSGraphTensorData *data = [[[MPSGraphTensorData alloc]
        initWithMTLBuffer:buffer
                    shape:[placeholder shape]
                 dataType:MPSDataTypeFloat32] autorelease];
    return MPSTensor{placeholder, data};
}

template <>
Tensor conv2d::mps<float>(Tensor input, const Tensor &kernel,
                          const std::optional<const Tensor> &bias,
                          const uint padding_h, const uint padding_w,
                          const uint stride_h, const uint stride_w,
                          const uint dilation_h, const uint dilation_w,
                          const PaddingMode padding_mode, uint groups) {
    ensure_same_type(input, kernel, bias);
    errs::bail_if(padding_mode != PaddingMode::Zeros,
                  "padding mode must be zeroes");
    errs::bail_if(groups != 1, "groups must be 1");

    const uint output_channels = kernel.output_channels();
    const uint output_h =
        output_hw_for_2d<float>(input.height(), kernel.height(), padding_h,
                                dilation_h, stride_h, false);
    const uint output_w = output_hw_for_2d<float>(
        input.width(), kernel.width(), padding_w, dilation_w, stride_w, false);

    uint num_samples;
    Tensor output;
    if (input.batched(sample_dims::CONV2D)) {
        num_samples = input.batch_size(sample_dims::CONV2D);
        output = Tensor({num_samples, output_channels, output_h, output_w},
                        input.scalar_type);
    } else {
        num_samples = 1;
        output =
            Tensor({output_channels, output_h, output_w}, input.scalar_type);
    }

    @autoreleasepool {
        MPSGraphDevice *device = (MPSGraphDevice *)gen_mps_graph_device();
        // MPSGraph *graph = (MPSGraph *)Context::mps_graph();
        MPSGraph *graph = [[MPSGraph new] autorelease];
        MPSGraphConvolution2DOpDescriptor *conv_desc =
            [[MPSGraphConvolution2DOpDescriptor alloc] autorelease];
        conv_desc.strideInY = stride_h;
        conv_desc.strideInX = stride_w;
        conv_desc.dilationRateInY = dilation_h;
        conv_desc.dilationRateInX = dilation_w;
        conv_desc.paddingTop = padding_h;
        conv_desc.paddingBottom = padding_h;
        conv_desc.paddingLeft = padding_w;
        conv_desc.paddingRight = padding_w;
        conv_desc.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
        conv_desc.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
        conv_desc.groups = 1;

        MPSTensor in_tens = feed_tensor(graph, device, input);

        MPSTensor kern_tens = feed_tensor(graph, device, kernel);
        MPSGraphTensor *output_tensor =
            [graph convolution2DWithSourceTensor:in_tens.placeholder
                                   weightsTensor:kern_tens.placeholder
                                      descriptor:conv_desc
                                            name:nil];

        std::optional<MPSTensor> bias_tens = std::nullopt;
        const bool has_bias = bias.has_value();
        if (has_bias) {
            bias_tens = feed_tensor(graph, device, *bias, true);
            output_tensor =
                [graph additionWithPrimaryTensor:output_tensor
                                 secondaryTensor:bias_tens->placeholder
                                            name:nil];
        }
        MPSGraphTensorData *output_data =
            output_tensor_data(device, output_tensor, output);

        NSMutableDictionary *feeds =
            [[[NSMutableDictionary alloc] initWithCapacity:3] autorelease];
        feeds[in_tens.placeholder] = in_tens.data;
        feeds[kern_tens.placeholder] = kern_tens.data;

        if (has_bias) {
            feeds[bias_tens->placeholder] = bias_tens->data;
        }

        id<MTLCommandQueue> command_queue =
            [[[device metalDevice] newCommandQueue] autorelease];
        MPSCommandBuffer *command_buffer =
            [MPSCommandBuffer commandBufferFromCommandQueue:command_queue];

        MPSGraphExecutionDescriptor *exec_desc =
            [MPSGraphExecutionDescriptor new];

        [graph encodeToCommandBuffer:command_buffer
                               feeds:feeds
                    targetOperations:nil
                   resultsDictionary:@{output_tensor : output_data}
                 executionDescriptor:exec_desc];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
    return output;
}

template <>
Tensor conv2d::mps<double>(Tensor input, const Tensor &kernel,
                           const std::optional<const Tensor> &bias,
                           const uint padding_h, const uint padding_w,
                           const uint stride_h, const uint stride_w,
                           const uint dilation_h, const uint dilation_w,
                           const PaddingMode padding_mode, uint groups) {
    errs::mps_metal_unsupported_double();

    Tensor input_float = input.template to_type<float>(ScalarType::Float32);
    Tensor kernel_float = kernel.template to_type<float>(ScalarType::Float32);
    std::optional<const Tensor> bias_float =
        bias.has_value() ? std::optional<Tensor>(bias->template to_type<float>(
                               ScalarType::Float32))
                         : std::nullopt;
    return conv2d::mps<float>(std::move(input_float), kernel_float, bias_float,
                              padding_h, padding_w, stride_h, stride_w,
                              dilation_h, dilation_w, padding_mode, groups)
        .template to_type<double>(ScalarType::Float64);
}
