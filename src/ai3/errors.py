# SPDX-License-Identifier: Apache-2.0

from typing import NoReturn
import sys


def print_error(mes):
    print(mes, file=sys.stderr)


def bail(message) -> NoReturn:
    raise AssertionError(message)


class UnsupportedCallableError(Exception):
    def __init__(self, module: str):
        super().__init__(
            f'unsupported callable: {module}')


def unsupported_mod(module) -> NoReturn:
    raise UnsupportedCallableError(str(module))


def bail_if(check, message):
    if check:
        bail(message)
