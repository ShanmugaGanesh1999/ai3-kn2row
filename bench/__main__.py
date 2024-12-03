from bench import layer, swap_conv2d, convert
import model_zoo

layer.run()
model_zoo.run_on(swap_conv2d.runner)
model_zoo.run_on(convert.runner)
