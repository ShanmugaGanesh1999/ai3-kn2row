# SPDX-License-Identifier: Apache-2.0

from . import _core, utils, errors


class Tensor():
    """Simple type which implements the
    `Python Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_"""

    def __init__(self, tens: _core.Tensor):
        self.core = tens

    def to(self, out_type):
        """
        Transform this Tensor to another type using
        the `buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.

        Args:
            out_type : type to transform this object to

        Returns:
            An object of *out_type* with the shape and contents of the original
        """
        if out_type is None:
            return self

        if not isinstance(out_type, str):
            out_type = utils.smart_type_str(out_type)

        if out_type == utils.TORCH_TENSOR_TYPE_STR:
            return self.torch()
        elif out_type == 'numpy.ndarray':
            return self.numpy()
        errors.bail(
            f'unsupported type to transfer tensor to {out_type}')

    def numpy(self):
        """
        Transform this Tensor to a *numpy.ndarray* using
        the `buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.

        Returns:
            *numpy.ndarray* with the shape and contents of the original
        """
        import numpy
        dtype = {
            _core.ScalarType(_core.ScalarType.Float32): numpy.float32,
            _core.ScalarType(_core.ScalarType.Float64): numpy.float64
        }[self.core.scalar_type]
        errors.bail_if(
            dtype is None,
            f'tensor type, {self.core.scalar_type} is neither float32 or float64')
        data: numpy.ndarray = numpy.frombuffer(
            self.core, dtype=dtype)
        return data.reshape(self.core.shape)

    def torch(self):
        """
        Transform this Tensor to a `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_ using
        the `buffer protocol <https://docs.python.org/3/c-api/buffer.html>`_.


        Returns:
            `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_ with the shape and contents of the original
        """
        import torch
        dtype = {
            _core.ScalarType(_core.ScalarType.Float32): torch.float32,
            _core.ScalarType(_core.ScalarType.Float64): torch.float64
        }[self.core.scalar_type]
        return torch.frombuffer(
            self.core, dtype=dtype).view(self.core.shape)
