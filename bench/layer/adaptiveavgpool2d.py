import torch
from bench import predict_show_time
from torch import nn
import ai3
from test import compare_tensors


class AdaptiveAvgPool2D(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2D, self).__init__()
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(
            output_size)

    def forward(self, x):
        x = self.adaptive_avgpool(x)
        return x


def run():
    print('AdaptiveAvgPool2D')
    input = torch.randn(1000, 3, 300, 300)
    orig = AdaptiveAvgPool2D(output_size=(50, 50))
    optim = ai3.convert(orig)
    orig_out = predict_show_time(
        orig, input, 'pytorch')
    assert (isinstance(orig_out, torch.Tensor))
    optim_out = predict_show_time(
        optim, input, 'ai3')
    compare_tensors(optim_out, orig_out.detach(
    ).numpy(), print_pass=False)


if __name__ == '__main__':
    run()
