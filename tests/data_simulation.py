"""
Simulated I-Q Signal Generation for Testing

This module provides functionality to simulate I-Q signals for unit testing.
Each single shot is a 1D complex array representing I (in-phase) and Q (quadrature) components over time `t`.

I and Q signals are derived by mixing the readout signal with a local oscillator and digitizing via ADCs. 
In this simulation, the '1' qubit state is represented by a fixed π phase shift, though in real-world data, this phase shift is unknown.

Simple noise is added to mimic real conditions.

For more details, see 'V. QUBIT READOUT' in 'A quantum engineer's guide to superconducting qubits' Appl. Phys. Rev. 6, 021318 (2019); https://doi.org/10.1063/1.5089550
"""

from typing import Iterable
import numpy as np
from qubit_measurement_analysis.data import SingleShot


def get_simulated_singleshot_instance(
    qubit_state: dict, qubit_freq: dict, time: Iterable
) -> SingleShot:

    assert qubit_state.keys() == qubit_freq.keys()

    iq_data_container = np.zeros(len(time), dtype=np.complex64)
    state_regs_container = {}
    for qubit, freq, state in zip(
        qubit_freq.keys(), qubit_freq.values(), qubit_state.values()
    ):
        global_phase = 2 * np.pi * freq * time
        amp_0 = 750
        amp_1 = 1000

        noise_bound_low = 100
        noise_bound_high = 200
        amp_normal_noise = np.random.normal(
            noise_bound_low, noise_bound_high, size=(len(time))
        ) + 1j * np.random.normal(noise_bound_low, noise_bound_high, size=(len(time)))

        random_phase_offset = np.random.normal(0, np.pi / 3)

        if state == "1":
            iq_data = (amp_1) * np.exp(
                1j * (global_phase + np.pi + random_phase_offset)
            ) + amp_normal_noise
        elif state == "0":
            iq_data = (amp_0) * np.exp(
                1j * (global_phase + random_phase_offset)
            ) + amp_normal_noise
        else:
            raise ValueError("state should be either '0' or '1'")

        iq_data_container += iq_data
        state_regs_container[qubit] = state

    return SingleShot(iq_data_container, state_regs_container)
