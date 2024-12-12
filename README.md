# Setup Instructions

Follow these steps to set up your environment:

1. Move into the project directory:

   ```sh
   cd ai3-kn2row
   ```

2. Execute the shell script:

   ```sh
   sh execute_kn2row.sh
   ```

   This script takes care of everything.

---

```cpp
// original src/ai3/custom/conv2d.hpp
#pragma once

#include <ai3.hpp>

/**
 * @DEFAULT_BOOL{Conv2D}
 */
const bool DEFAULT_CONV2D = false;

/**
 * @CUSTOM_OP{Conv2D,conv2d}
 */
template <typename dtype>
Tensor conv2d_custom(Tensor input, const Tensor &kernel,
                const std::optional<const Tensor> &bias,
                const uint padding_h, const uint padding_w,
                const uint stride_h, const uint stride_w,
                const uint dilation_h, const uint dilation_w,
                const PaddingMode padding_mode, uint groups) {
   errs::no_user_def("conv2d");
}
```
