import torch
import ai3
import torch.nn.functional as F


class TestModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TestModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        return x


def get_grad(model, input, target):
    out = model(input)
    loss = F.mse_loss(out, target)
    model.zero_grad()
    loss.backward(retain_graph=True)
    return {name: param.grad.clone() for name, param in model.named_parameters()}


def conv2d():
    test_conv2d_with(torch.randn(
        10, 300, 300), 32, 3, 'no batch')
    test_conv2d_with(torch.randn(
        1, 3, 224, 224), 16, (4, 3), 'batch = 1')
    test_conv2d_with(torch.randn(
        10, 10, 512, 52), 5, 5, 'batch = 10')


def test_conv2d_with(input, out_channels, kernel_size, mes):
    input.requires_grad = True
    model = TestModel(
        input.shape[len(input.shape) - 3], out_channels, kernel_size)
    out_shape = model(input).shape
    target = torch.randn(out_shape)

    grad_torch = get_grad(model, input, target)
    ai3.swap_conv2d(model)
    grad_ai3 = get_grad(model, input, target)

    same_gradients = True
    for name in grad_torch:
        if not torch.allclose(grad_torch[name], grad_ai3[name]):
            print(
                f'Gradients for {name} on {mes} differ')
            same_gradients = False

    if same_gradients:
        print(
            f'Gradients are the same for conv2d on {mes}')
    else:
        print(
            f'Gradients are different for conv2d on {mes}')
