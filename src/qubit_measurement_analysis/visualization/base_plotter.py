"Basic visualization functionality"
from typing import Union, Iterable
from matplotlib import axes
import numpy as np

from qubit_measurement_analysis.visualization.utils import _get_current_kwargs


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
        current_kwargs = _get_current_kwargs(kwargs, 0)
        in_phase = ax.plot(
            x, value.real.flatten(), label=f"in-phase-reg({label})", **current_kwargs
        )
        current_kwargs = _get_current_kwargs(kwargs, 1)
        quadrature = ax.plot(
            x,
            value.imag.flatten(),
            label=f"quadrature-reg({label})",
            **current_kwargs,
        )
        return in_phase, quadrature
