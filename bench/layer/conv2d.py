import torch
from bench import predict_show_time
from torch import nn
import ai3
from test import compare_tensors
from run import CONV2D_ALGOS_TO_USE

N = 100


class Conv2D(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


def perform_pred_with(algo, orig, input):
    input_shape = tuple(input.size())
    optim = ai3.convert(
        orig, {'conv2d': algo})
    return predict_show_time(
        optim, input, f'ai3 {algo} {input_shape}')


def run_on(input):
    orig = Conv2D(
        input.shape[1], input.shape[1], (3, 3))
    input_shape = tuple(input.size())

    orig_out = predict_show_time(
        orig, input, f'pytorch {input_shape}')
    assert (isinstance(orig_out, torch.Tensor))

    for algo in CONV2D_ALGOS_TO_USE:
        out = perform_pred_with(algo, orig, input)
        if algo == 'metal':
            atol = 1e-1
        else:
            atol = None
        compare_tensors(out, orig_out, algo, print_pass=False, atol=atol)


print('Conv2D')
run_on(torch.randn(N, 3, 224, 224))
print('-------------')
run_on(torch.randn(N, 512, 14, 14))
print('-------------')
run_on(torch.randn(N, 32, 7, 7))
