"Basic visualization functionality"
from typing import Union, Iterable
from matplotlib import axes
import numpy as np


class BasicShotPlotter:
    """Class for generating basic plots."""

    @staticmethod
    def scatter_matplotlib(
        ax: axes.Axes,
        value: Iterable,
        label: Union[str, Iterable],
        marker: Union[str, Iterable],
        **kwargs,
    ):
        """Generate a basic scatter plot using Matplotlib."""
        scatter = ax.scatter(
            x=value.real, y=value.imag, label=label, marker=marker, **kwargs
        )
        return scatter

    @staticmethod
    def plot_matplotlib(
        ax: axes.Axes,
        value: Iterable,
        label: Union[str, Iterable],
        x: Iterable = None,
        **kwargs,
    ):
        """Generate a basic line plot using Matplotlib"""
        x = x if x is not None else np.arange(value.shape[-1])
        in_phase = ax.plot(
            x, value.real.flatten(), label=f"in-phase-reg({label})", **kwargs
        )
        quadrature = ax.plot(
            x, value.imag.flatten(), label=f"quadrature-reg({label})", **kwargs
        )
        return in_phase, quadrature
