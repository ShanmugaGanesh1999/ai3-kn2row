import time
import ai3
import torch


def warm_up(runner, data):
    data_sample = data[0]
    data_shape = (1,) + data_sample.size()
    runner(data_sample.view(data_shape))


def timed_predict(runner, data, grad: bool = False):
    out = None
    start_time = None
    if isinstance(runner, torch.nn.Module):
        warm_up(runner, data)
        if grad:
            start_time = time.time()
            out = runner(data)
        else:
            with torch.inference_mode():
                start_time = time.time()
                out = runner(data)
    elif isinstance(runner, ai3.Model):
        warm_up(runner, data)
        start_time = time.time()
        out = runner.predict(
            data, out_type=torch.Tensor)
    else:
        print(f'invalid runner f{type(runner)}')
        assert (False)
    end_time = time.time()
    assert (start_time > 0)
    latency = end_time - start_time

    return out, latency


def predict_show_time(runner, data, runner_name: str, grad: bool = False):
    out, latency = timed_predict(runner, data, grad=grad)
    print(
        f'  Time {runner_name}: {latency} seconds')
    return out
