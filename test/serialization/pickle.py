import torch
import model_zoo
import sys
import os
import pickle
from test import compare_tensors
import ai3
from ai3.errors import UnsupportedCallableError


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    try:
        target = module(input_data)
        with open(f'torch.pkl', 'wb') as f:
            pickle.dump(module, f)
        with open(f'torch.pkl', 'rb') as f:
            unpickled_torch_module = pickle.load(f)
        target_unpickled = unpickled_torch_module(input_data)
        compare_tensors(
            target_unpickled, target,
            f'{name} torch orig and torch unpickled, {model_zoo.BATCH} samples')

        try:
            ai3_model = ai3.convert(module)
        except UnsupportedCallableError as e:
            print(f'  {e} so skipping')
            return
        sb_out = ai3_model(input_data)
        compare_tensors(
            sb_out, target,
            f'{name} swap backend, not pickled, {model_zoo.BATCH} samples')
        ai3.swap_conv2d(module)
        sc_out = module(input_data)
        compare_tensors(
            sc_out, target,
            f'{name} swap conv2d, not pickled, {model_zoo.BATCH} samples')

        with open(f'ai3_sc.pkl', 'wb') as f:
            pickle.dump(module, f)
        with open(f'ai3_sc.pkl', 'rb') as f:
            unpickled_ai3_swapped = pickle.load(f)

        output_unpickled_sc = unpickled_ai3_swapped(input_data)
        compare_tensors(
            output_unpickled_sc, target_unpickled,
            f'{name} torch unpickled and ai3 swap unpickled, {model_zoo.BATCH} samples')

        with open(f'ai3_sb.pkl', 'wb') as f:
            pickle.dump(ai3_model, f)
        with open(f'ai3_sb.pkl', 'rb') as f:
            unpickled_ai3_model = pickle.load(f)

        output_unpickled_sb = unpickled_ai3_model(input_data)

        compare_tensors(
            output_unpickled_sb, target_unpickled,
            f'{name} torch unpickled and ai3 model unpickled, {model_zoo.BATCH} samples')
    finally:
        os.remove('torch.pkl')
        os.remove('ai3_sc.pkl')
        os.remove('ai3_sb.pkl')


if __name__ == '__main__':
    model_zoo.from_args(runner, sys.argv)
