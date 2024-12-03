import torch
from bench import predict_show_time
from torch import nn
import ai3
from test import compare_tensors


class MaxPool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2D, self).__init__()
        self.maxpool = nn.MaxPool2d(
            kernel_size, stride, padding)

    def forward(self, x):
        x = self.maxpool(x)
        return x


print('MaxPool2D')
input = torch.randn(1000, 3, 300, 300)
orig = MaxPool2D(
    kernel_size=5, stride=1, padding=0)
optim = ai3.convert(orig)
orig_out = predict_show_time(
    orig, input, 'pytorch')
assert (isinstance(orig_out, torch.Tensor))
optim_out = predict_show_time(optim, input, 'ai3')
compare_tensors(optim_out, orig_out.detach(
).numpy(), print_pass=False)
