"Single-Shot plotting functionality"
from typing import Iterable
from matplotlib import axes, pyplot as plt
from qubit_measurement_analysis.visualization.base_plotter import (
    BasicShotPlotter as bsp,
)
from qubit_measurement_analysis.visualization.utils import _get_current_kwargs


class SingleShotPlotter:
    # TODO: describe functionality of the class

    def __init__(self, children) -> None:
        self.children = children

    def scatter(self, ax: axes.Axes = None, **kwargs):
        # TODO: add docstring
        # TODO: add notion about passing iterable kwargs
        kwargs_ = kwargs.copy()
        q_registers = (
            self.children.q_registers
            if self.children.is_demodulated
            else [self.children.q_registers]
        )

        if ax is None:
            _, ax = plt.subplots()

        for reg_idx, qubit in enumerate(q_registers):
            if kwargs.get("marker") is None:
                marker = (
                    f"${self.children.state[reg_idx : len(qubit) + reg_idx]}$"
                    if self.children.is_demodulated
                    else None
                )
                kwargs_.update({"marker": marker})

            if kwargs.get("label") is None:
                kwargs_.update({"label": qubit})

            current_kwargs = _get_current_kwargs(kwargs_, reg_idx)
            _ = bsp.scatter_matplotlib(
                ax,
                self.children.value[reg_idx, :],
                **current_kwargs,
            )
        return ax

    def plot(self, ax: axes.Axes = None, x: Iterable = None, **kwargs):
        # TODO: add docstring
        # TODO: add notion about passing iterable kwargs
        kwargs_ = kwargs.copy()
        q_registers = (
            self.children.q_registers
            if self.children.is_demodulated
            else [self.children.q_registers]
        )

        if ax is None:
            _, ax = plt.subplots()

        for reg_idx, qubit in enumerate(q_registers):
            if kwargs.get("label") is None:
                kwargs_.update({"label": [f"real({qubit})", f"imag({qubit})"]})
                current_kwargs = kwargs_.copy()
            else:
                current_kwargs = _get_current_kwargs(kwargs_, reg_idx)
            _ = bsp.plot_matplotlib(
                ax,
                self.children.value[reg_idx, :],
                x=x,
                **current_kwargs,
            )
        return ax
