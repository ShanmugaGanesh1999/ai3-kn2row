import torch
import ai3
import model_zoo
import sys
from test import compare_tensors
from run import CONV2D_ALGOS_TO_USE


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    with torch.inference_mode():
        for algo in CONV2D_ALGOS_TO_USE:
            ai3.swap_conv2d(module, algo)
            output = module(input_data)
            compare_tensors(
                output, target,
                f'{name} swap conv2d with {algo}, {model_zoo.BATCH} samples')


if __name__ == '__main__':
    model_zoo.from_args(runner, sys.argv)
