import torch
from torch import nn
import ai3  # the framework


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        return x


input_data = torch.randn(10, 3, 224, 224)
orig = ConvNet()
torch_out = orig(input_data)
model: ai3.Model = ai3.swap_backend(orig,
                                    {"conv2d": "direct"})
sb_out = model(input_data)
ai3.swap_conv2d(orig, ["direct", "smm"])
sc_out = orig(input_data)
assert torch.allclose(torch_out, sb_out, atol=1e-6)
assert torch.allclose(torch_out, sc_out, atol=1e-6)
