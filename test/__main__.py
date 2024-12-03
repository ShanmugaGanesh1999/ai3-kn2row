from test import unit, convert, swap_conv2d, ops
import model_zoo

unit.run()
model_zoo.run_on(swap_conv2d.runner)
model_zoo.run_on(convert.runner)
ops.run()
