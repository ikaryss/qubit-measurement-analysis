"Functional tools for visualization"
from typing import Iterable


def _get_current_kwargs(kwargs: dict, iteration: int):
    # TODO: write docstring
    current_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, Iterable) and not isinstance(value, str):
            current_kwargs[key] = value[iteration]
        else:
            current_kwargs[key] = value
    return current_kwargs
