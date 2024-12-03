import torch
import ai3
from test import compare_tensors
from test.ops.compile import compile
from bench import predict_show_time, timed_predict
import model_zoo
import sys
import statistics

N = 100


def show_stat_of(runner, input_data, name, *, N, grad):
    times = []
    out = None
    for _ in range(N):
        (out, time) = timed_predict(
            runner, input_data, grad=grad)
        times.append(time)
    med = statistics.median(times)
    print(f'  Median of {N} runs for {name}: {med}')
    minimum = min(times)
    print(f'  Minimum of {N} runs for {name}: {minimum}')
    maximum = max(times)
    print(f'  Maximum of {N} runs for {name}: {maximum}')
    return out


def runner(module: torch.nn.Module, input_data: torch.Tensor, name: str):
    predict_show_time(module, input_data, name + ' torch orig')
    torch_comp = compile(module)

    ai3.swap_conv2d(module)
    predict_show_time(module, input_data, name + ' ai3 orig')
    ai3_comp = compile(module)

    target = show_stat_of(torch_comp, input_data,
                          name + ' torch comp grad', N=N, grad=True)

    output = show_stat_of(ai3_comp, input_data,
                          name + ' ai3 comp grad',  N=N, grad=True)

    compare_tensors(
        output, target,
        f'{name} ai3 comp with grad, {model_zoo.BATCH} samples',
        print_pass=False)

    target = show_stat_of(torch_comp, input_data,
                          name + ' torch comp no grad', N=N, grad=False)
    output = show_stat_of(ai3_comp, input_data,
                          name + ' ai3 comp no grad', N=N, grad=False)

    compare_tensors(
        output, target, f'{name} ai3 comp no grad, {model_zoo.BATCH} samples',
        print_pass=False)


if __name__ == '__main__':
    model_zoo.from_args(runner, sys.argv)
