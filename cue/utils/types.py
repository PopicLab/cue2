from enum import Enum
from collections import defaultdict


class NestedDict(defaultdict):
    def __call__(self):
        return NestedDict(self.default_factory)


def make_enum(enum_name, enum_values, module):
    enum = Enum(enum_name, {value: value for value in enum_values}, type=str)
    enum.__module__ = module
    return enum

