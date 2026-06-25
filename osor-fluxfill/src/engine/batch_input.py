"""Shared batch-input container used by trainers and batch-inference scripts.

A BatchInput is a lightweight attribute bag populated once per step. The
``__setattr__`` guard catches accidental double-assignment (a common bug when
merging per-stage fields); ``update`` is the intentional escape hatch.
"""


class BatchInput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ValueError(f"Duplicated key in BatchInput: {name}")
        self.__dict__[name] = value

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.__dict__[name] = value
