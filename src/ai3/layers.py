# SPDX-License-Identifier: Apache-2.0

from typing import (
    Union,
    Sequence,
    Optional,
    Tuple
)
from abc import ABC, abstractmethod
from . import _core, errors, utils


class Layer(ABC):
    def __init__(self, core):
        self.core = core
        ...

    def algo(self) -> str:
        return self.core.algorithm


class Conv2D(Layer):
    def __init__(
            self,  weight, bias, _stride: Union[int, Tuple[int, ...]],
            padding: Union[str, Union[int, Tuple[int, ...]]],
            _dilation: Union[int, Tuple[int, ...]],
            padding_mode: Union[str, int, _core.PaddingMode],
            groups: int, algorithm: str, scalar_type: _core.ScalarType):
        stride = utils.make_2d(_stride)
        dilation = utils.make_2d(_dilation)
        padding = utils.make_padding_2d(
            padding, stride, dilation, weight.size())
        if isinstance(padding_mode, str):
            errors.bail_if(
                padding_mode
                not in ['zeros', 'reflect', 'replicate', 'circular'],
                f'invalid padding mode: {padding_mode}')
            padding_mode = _core.PaddingMode({
                'zeros': _core.PaddingMode.zeros,
                'reflect': _core.PaddingMode.reflect,
                'replicate': _core.PaddingMode.replicate,
                'circular': _core.PaddingMode.circular
            }[padding_mode])

        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        self.core = _core.Conv2D(
            weight_addr, weight_shape, bias_addr, padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
            _core.PaddingMode(padding_mode),
            groups, algorithm, scalar_type)


class Linear(Layer):
    def __init__(self, weight, bias, algorithm: str,
                 scalar_type: _core.ScalarType):
        weight_addr = utils.get_address(weight)
        weight_shape = utils.get_shape(weight)
        if bias is not None:
            bias_addr = utils.get_address(bias)
        else:
            bias_addr = None
        self.core = _core.Linear(
            weight_addr, weight_shape, bias_addr, algorithm, scalar_type)


class ReLU(Layer):
    def __init__(self, algorithm: str):
        self.core = _core.ReLU(algorithm)


class MaxPool2D(Layer):
    def __init__(self, kernel_shape: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 dilation: Union[int, Tuple[int, int]],
                 ceil_mode: bool,
                 algorithm: str):
        stride = utils.make_2d(stride)
        dilation = utils.make_2d(dilation)
        kernel_shape = utils.make_2d(kernel_shape)
        padding = utils.make_2d(padding)
        ceil_mode = ceil_mode

        self.core = _core.MaxPool2D(
            kernel_shape[0],
            kernel_shape[1],
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
            ceil_mode, algorithm)


class AvgPool2D(Layer):
    def __init__(self, kernel_shape: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 ceil_mode: bool,
                 count_include_pad: bool,
                 divisor_override: Optional[int],
                 algorithm: str):
        padding_2d = utils.make_2d(padding)
        stride_2d = utils.make_2d(stride)
        kernel_shape = utils.make_2d(kernel_shape)

        self.core = _core.AvgPool2D(
            kernel_shape[0],
            kernel_shape[1],
            padding_2d[0],
            padding_2d[1],
            stride_2d[0],
            stride_2d[1],
            ceil_mode, count_include_pad, divisor_override, algorithm)


class AdaptiveAvgPool2D(Layer):
    def __init__(
            self,
            output_shape: Optional[Union[int, Sequence[Optional[int]]]],
            algorithm: str):
        if isinstance(output_shape, int):
            output_shape = utils.make_2d(
                output_shape)
        elif output_shape is None:
            output_shape = [None, None]
        self.core = _core.AdaptiveAvgPool2D(
            output_shape[0],
            output_shape[1],
            algorithm)


class Flatten(Layer):
    def __init__(self, start_dim: int, end_dim: int, algorithm: str):
        self.core = _core.Flatten(
            start_dim, end_dim, algorithm)
