import torch
import torchvision
import ai3


class SpecialConv(torch.nn.Module):
    def __init__(self, orig: torch.nn.Conv2d, algorithm: str):
        super(SpecialConv, self).__init__()
        self.orig = orig
        self.algorithm = algorithm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.orig(x)


def conv2d_selector(orig: torch.nn.Conv2d) -> str:
    in_channels = orig.weight.shape[1]
    if in_channels > 200:
        return 'smm'
    return 'direct'


input_data = torch.randn(1, 3, 224, 224)
vgg16 = torchvision.models.vgg16(
    weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16.eval()
with torch.inference_mode():
    torch_out = vgg16(input_data)

    ai3.swap_conv2d(
        vgg16, conv2d_selector, None, swap_with=SpecialConv)
    sb_out = vgg16(input_data)
    assert torch.allclose(
        torch_out, sb_out, atol=1e-4)
