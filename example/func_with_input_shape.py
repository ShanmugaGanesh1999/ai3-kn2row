import torch
import ai3
from ai3 import swap_torch
from .manual_conv2d import ConvNet
from typing import Sequence, Union


def conv2d_selector(
        orig: Union[torch.nn.Conv2d, swap_torch.Conv2D],
        input_shape: Sequence[int]) -> str:
    out_channels = orig.weight.shape[0]
    if (out_channels < 50 and
        input_shape[0] < 50 and
        input_shape[1] > 150 and
            input_shape[2] > 150):
        return 'direct'
    return 'smm'


input_data = torch.randn(10, 3, 224, 224)
orig = ConvNet()

with torch.inference_mode():
    torch_out = orig(input_data)
    ai3.swap_conv2d(
        orig, conv2d_selector, (3, 224, 224))
    sc_out_first = orig(input_data)

    ai3.swap_conv2d(
        orig, conv2d_selector, (3, 512, 512))
    sc_out_second = orig(input_data)

    assert torch.allclose(
        torch_out, sc_out_first, atol=1e-6)
    assert torch.allclose(
        torch_out, sc_out_second, atol=1e-6)
    model = ai3.convert(
        orig, {'conv2d': conv2d_selector, 'maxpool2d': 'default'},
        (3, 224, 224))
    sb_out = model.predict(input_data)
    assert isinstance(sb_out, ai3.Tensor)
    assert torch.allclose(
        torch_out, sb_out.torch(), atol=1e-6)
