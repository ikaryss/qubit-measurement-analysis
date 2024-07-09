"Single-Shot plotting functionality"
from typing import Iterable
from matplotlib import axes, pyplot as plt
from qubit_measurement_analysis.visualization.base_plotter import (
    BasicShotPlotter as bsp,
)


class SigngleShotPlotter:
    # TODO: describe functionality of the class

    def __init__(self, children) -> None:
        self.children = children

    def scatter(self, ax: axes.Axes = None, **kwargs):
        # TODO: add docstring
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
            _ = bsp.scatter_matplotlib(
                ax,
                self.children.value[reg_idx, :],
                label=qubit,
                marker=marker,
                **kwargs,
            )
        return ax

    def plot(self, ax: axes.Axes = None, x: Iterable = None, **kwargs):
        # TODO: add docstring
        q_registers = (
            self.children.q_registers
            if self.children.is_demodulated
            else [self.children.q_registers]
        )

        if ax is None:
            _, ax = plt.subplots()

        for reg_idx, qubit in enumerate(q_registers):

            _ = bsp.plot_matplotlib(
                ax,
                self.children.value[reg_idx, :],
                label=qubit,
                x=x,
                **kwargs,
            )
        return ax
