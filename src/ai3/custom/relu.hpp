#pragma once

#include <ai3.hpp>

/**
 * @DEFAULT_BOOL{ReLU}
 */
const bool DEFAULT_RELU = false;

/**
 * @CUSTOM_OP{ReLU,relu}
 */
template <typename dtype> Tensor relu_custom(Tensor input) {
    errs::no_user_def("relu");
}
