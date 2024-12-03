import torch
from torch import nn
import ai3
import platform

PASS_MES = 'ai3 and torch Models compiled with torch.compile produce same outputs '


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1,
            bias=False)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        return x


def compile(orig):
    if platform.system() == 'Darwin':
        return torch.compile(orig, backend='aot_eager')
    else:
        return torch.compile(orig)


def conv2d():
    input_data = torch.randn(3, 224, 224)
    orig = ConvNet()
    tar = orig(input_data)

    ai3.swap_conv2d(orig, 'direct')
    swap_comped = compile(orig)
    swap_comped_out = swap_comped(input_data)

    assert torch.allclose(
        swap_comped_out, tar, atol=1e-6)
    print(PASS_MES + 'conv2d')
