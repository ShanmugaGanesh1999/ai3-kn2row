from . import backward, compile, opcheck, train


def run():
    opcheck.conv2d()
    compile.conv2d()
    backward.conv2d()
    train.conv2d()
