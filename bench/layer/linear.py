import torch
from bench import predict_show_time
from torch import nn
import ai3
from test import compare_tensors


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(
            input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


print('Linear')
input = torch.randn(1000, 1200)
orig = Linear(1200, 800)
sb = ai3.convert(orig)
sb_out = predict_show_time(sb, input, 'ai3')
print(f'type: {type(sb_out)}')
orig_out = predict_show_time(
    orig, input, 'pytorch')
compare_tensors(sb_out, orig_out, print_pass=False)
