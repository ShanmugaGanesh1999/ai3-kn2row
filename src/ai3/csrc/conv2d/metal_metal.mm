// SPDX-License-Identifier: Apache-2.0

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algos.hpp>

inline uint computeMPSAlignOffset(const uint kernel, const uint padding,
                                  const uint dilation = 1) {
    return (((kernel - 1) * dilation + 1) / 2) - padding;
}

void *gen_mtl_device() {
    id<MTLDevice> *ret = new id<MTLDevice>;
    *ret = MTLCreateSystemDefaultDevice();
    return ret;
}

void release_mtl_device(id<MTLDevice> *dev) { delete dev; }

template <typename dtype> Tensor kernel_to_ohwi(const Tensor &orig) {
    const uint out_channels = orig.output_channels();
    const uint in_channels = orig.input_channels();
    const uint kern_w = orig.width();
    const uint kern_h = orig.height();
    std::vector<uint> new_shape(4);
    new_shape[0] = out_channels;
    new_shape[1] = kern_h;
    new_shape[2] = kern_w;
    new_shape[3] = in_channels;
    Tensor kern(new_shape, orig.scalar_type);
    dtype *kern_data = data_as<dtype>(kern.data);
    const dtype *orig_data = data_as<dtype>(orig.data);
    for (size_t o = 0; o < out_channels; ++o) {
        for (size_t i = 0; i < in_channels; ++i) {
            for (size_t h = 0; h < kern_h; ++h) {
                for (size_t w = 0; w < kern_w; ++w) {
                    uint orig_id =
                        ((o * in_channels + i) * kern_h + h) * kern_w + w;
                    uint new_id =
                        ((o * kern_h + h) * kern_w + w) * in_channels + i;
                    kern_data[new_id] = orig_data[orig_id];
                }
            }
        }
    }
    return kern;
}

template <typename dtype> Tensor kernel_to_hwio(const Tensor &orig) {
    const uint out_channels = orig.output_channels();
    const uint in_channels = orig.input_channels();
    const uint kern_w = orig.width();
    const uint kern_h = orig.height();
    std::vector<uint> new_shape(4);
    new_shape[0] = kern_h;
    new_shape[1] = kern_w;
    new_shape[2] = in_channels;
    new_shape[3] = out_channels;
    Tensor kern = Tensor(new_shape, orig.scalar_type);
    const dtype *orig_data = data_as<dtype>(orig.data);
    dtype *new_data = data_as<dtype>(kern.data);
    for (size_t o = 0; o < out_channels; ++o) {
        for (size_t i = 0; i < in_channels; ++i) {
            for (size_t h = 0; h < kern_h; ++h) {
                for (size_t w = 0; w < kern_w; ++w) {
                    uint orig_id =
                        ((o * in_channels + i) * kern_h + h) * kern_w + w;
                    uint new_id =
                        ((h * kern_w + w) * in_channels + i) * out_channels + o;

                    new_data[new_id] = orig_data[orig_id];
                }
            }
        }
    }
    return kern;
}

template <>
Tensor conv2d::metal<float>(Tensor input, const Tensor &kernel,
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
    const uint input_channels = kernel.input_channels();
    const uint output_h =
        output_hw_for_2d<float>(input.height(), kernel.height(), padding_h,
                                dilation_h, stride_h, false);
    const uint output_w = output_hw_for_2d<float>(
        input.width(), kernel.width(), padding_w, dilation_w, stride_w, false);

    const uint kernel_h = kernel.height();
    const uint kernel_w = kernel.width();
    const uint input_h = input.height();
    const uint input_w = input.width();

    Tensor kernel_ohwi = kernel_to_ohwi<float>(kernel);

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
        id<MTLDevice> *dev_ptr =
            static_cast<id<MTLDevice> *>(Context::mtl_device());
        id<MTLDevice> device = *dev_ptr;
        MPSImageDescriptor *input_desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:input_w
                                      height:input_h
                             featureChannels:input_channels];

        NSMutableArray *input_ims =
            [NSMutableArray arrayWithCapacity:num_samples];
        for (uint i = 0; i < num_samples; ++i) {
            MPSImage *input_im =
                [[[MPSImage alloc] initWithDevice:device
                                  imageDescriptor:input_desc] autorelease];
            [input_im writeBytes:(float *)input.data +
                                 i * input_channels * input_h * input_w
                      dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
                      imageIndex:0];
            [input_ims addObject:input_im];
        }

        MPSImageDescriptor *output_desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:output_w
                                      height:output_h
                             featureChannels:output_channels];
        NSMutableArray *output_ims =
            [NSMutableArray arrayWithCapacity:num_samples];
        for (uint i = 0; i < num_samples; ++i) {
            MPSImage *output_im =
                [[[MPSImage alloc] initWithDevice:device
                                  imageDescriptor:output_desc] autorelease];
            [output_ims addObject:output_im];
        }

        MPSCNNConvolutionDescriptor *conv_desc = [MPSCNNConvolutionDescriptor
            cnnConvolutionDescriptorWithKernelWidth:kernel_w
                                       kernelHeight:kernel_h
                               inputFeatureChannels:input_channels
                              outputFeatureChannels:output_channels];
        conv_desc.strideInPixelsX = stride_w;
        conv_desc.strideInPixelsY = stride_h;
        conv_desc.dilationRateX = dilation_w;
        conv_desc.dilationRateY = dilation_h;

        MPSOffset offset;
        offset.x = computeMPSAlignOffset(kernel_w, padding_w, dilation_w);
        offset.y = computeMPSAlignOffset(kernel_h, padding_h, dilation_h);
        offset.z = 0;

        MPSCNNConvolution *conv = [[[MPSCNNConvolution alloc]
                   initWithDevice:device
            convolutionDescriptor:conv_desc
                    kernelWeights:(float *)kernel_ohwi.data
                        biasTerms:bias.has_value() ? (float *)bias->data
                                                   : nullptr
                            flags:MPSCNNConvolutionFlagsNone] autorelease];
        [conv setEdgeMode:MPSImageEdgeModeZero];
        [conv setOffset:offset];

        id<MTLCommandQueue> command_queue =
            [[device newCommandQueue] autorelease];
        MPSCommandBuffer *command_buffer =
            [MPSCommandBuffer commandBufferFromCommandQueue:command_queue];

        [conv encodeBatchToCommandBuffer:command_buffer
                            sourceImages:input_ims
                       destinationImages:output_ims];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        for (uint i = 0; i < num_samples; i++) {
            [output_ims[i] readBytes:(float *)output.data +
                                     i * output_channels * output_h * output_w
                          dataLayout:MPSDataLayoutFeatureChannelsxHeightxWidth
                          imageIndex:0];
        }
    }
    return output;
}

template <>
Tensor conv2d::metal<double>(Tensor input, const Tensor &kernel,
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
    return conv2d::metal<float>(std::move(input_float), kernel_float,
                                bias_float, padding_h, padding_w, stride_h,
                                stride_w, dilation_h, dilation_w, padding_mode,
                                groups)
        .template to_type<double>(ScalarType::Float64);
}
