import torch
import ai3
from ai3.errors import UnsupportedCallableError
from run import CONV2D_ALGOS_TO_USE
from test import compare_tensors
from bench import predict_show_time
import model_zoo
import sys


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = predict_show_time(
        module, input_data, name + ' torch')
    for algo in CONV2D_ALGOS_TO_USE:
        try:
            ai3_model = ai3.convert(module, {'conv2d': algo})
        except UnsupportedCallableError as e:
            print(f'  {e} so skipping')
            return
        output = predict_show_time(
            ai3_model, input_data, f'{name} ai3 using {algo} conv2d')
        compare_tensors(
            output, target, f'{name} ai3, {model_zoo.BATCH} samples',
            print_pass=False)


if __name__ == '__main__':
    model_zoo.from_args(runner, sys.argv)
