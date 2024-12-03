import torch
from bench import predict_show_time
from torch import nn
import ai3
from test import compare_tensors


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return x


print('ReLU')
input = torch.randn(1000, 3, 1000, 500)
orig = ReLU()
optim = ai3.convert(orig)
orig_out = predict_show_time(
    orig, input, 'pytorch')
assert (isinstance(orig_out, torch.Tensor))
optim_out = predict_show_time(optim, input, 'ai3')
compare_tensors(optim_out, orig_out.detach(
).numpy(), '', print_pass=False)
