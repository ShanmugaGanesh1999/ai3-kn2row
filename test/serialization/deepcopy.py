import torch
import ai3
from ai3.errors import UnsupportedCallableError
import model_zoo
import sys
import ai3.swap_torch
from copy import deepcopy
from test import compare_tensors


def alter_padding(module, target_padding=(100, 100)):
    for c in module.children():
        if isinstance(c, (torch.nn.Conv2d, ai3.swap_torch.Conv2D)):
            assert c.padding != target_padding
            c.padding = target_padding
        elif len(list(c.children())) > 0:
            alter_padding(c, target_padding)


def alter_orig_after_copy_ensure_same(
        orig, input_data: torch.Tensor, mes: str):
    target = orig(input_data)
    cpy = deepcopy(orig)
    alter_padding(orig)
    cpy_out = cpy(input_data)
    compare_tensors(cpy_out, target, mes)
    return cpy_out


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    target = module(input_data)
    torch_cpy_out = alter_orig_after_copy_ensure_same(
        deepcopy(module), input_data, f'{name} torch after copying then altering orig')
    try:
        ai3_model = ai3.convert(module)
    except UnsupportedCallableError as e:
        print(f'  {e} so skipping')
        return
    sb_out = ai3_model(input_data)
    compare_tensors(
        sb_out, target,
        f'{name} swap backend, not copied, {model_zoo.BATCH} samples')

    ai3.swap_conv2d(module)
    sc_out = module(input_data)
    compare_tensors(
        sc_out, target,
        f'{name} swap conv2d, not copied, {model_zoo.BATCH} samples')

    ai3_sc_cpy_out = alter_orig_after_copy_ensure_same(
        module, input_data, f'{name} ai3 sc after copying then altering orig')
    compare_tensors(
        ai3_sc_cpy_out, torch_cpy_out,
        f'{name} torch copied and ai3 sc copied, {model_zoo.BATCH} samples')

    ai3_sb_cpy = deepcopy(ai3_model)
    del ai3_model.core
    ai3_sb_cpy_out = ai3_sb_cpy(input_data)
    compare_tensors(
        ai3_sb_cpy_out, torch_cpy_out,
        f'{name} torch copied and ai3 sb copied, {model_zoo.BATCH} samples')


if __name__ == '__main__':
    model_zoo.from_args(runner, sys.argv)
