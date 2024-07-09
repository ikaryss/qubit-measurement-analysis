"Single-Shot plotting functionality"
import numpy as np
from matplotlib import pyplot as plt
from qubit_measurement_analysis.visualization.base_plotter import (
    BasicShotPlotter as bsp,
)


class CollectionPlotter:
    # TODO: describe functionality of the class

    def __init__(self, children) -> None:
        self.children = children

    def scatter(self, ax: plt.Axes = None, **kwargs):

        if self.children.is_demodulated:
            q_registers = self.children.q_registers
            states = [state for state in np.sort(self.children.unique_states)]
        else:
            q_registers = [self.children.q_registers]
            states = np.sort(self.children.unique_states)

        if ax is None:
            _, ax = plt.subplots()

        for state in states:
            collection = self.children.filter_by_state(state)
            for reg_idx, reg in enumerate(q_registers):
                if kwargs.get("marker") is None:
                    marker = (
                        f"${state[reg_idx : len(reg) + reg_idx]}$"
                        if self.children.is_demodulated
                        else None
                    )
                _ = bsp.scatter_matplotlib(
                    ax,
                    collection.all_values[:, reg_idx, :],
                    label=reg,
                    marker=marker,
                    **kwargs,
                )
        return ax
