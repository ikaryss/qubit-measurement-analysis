"""Data processing functionality for Single-Shots.

This module provides the SingleShot class for storing and processing single-shot
measurement data in quantum experiments.
"""

import os
import uuid
import glob
from typing import Dict

import numpy as np
from numpy.typing import ArrayLike

from qubit_measurement_analysis.visualization.single_shot_plotter import (
    SingleShotPlotter as ssp,
)
from qubit_measurement_analysis import ArrayModule
from qubit_measurement_analysis._transformations import (
    _mean,
    _mean_filter,
    _mean_centring,
    _normalize,
    _standardize,
    _demodulate,
)

# DEFAULT_DTYPE: type = np.complex64


class SingleShot:
    """A class for storing and processing a single SingleShot entity.

    This class represents a single-shot measurement in quantum experiments,
    providing methods for data manipulation, analysis, and visualization.

    This is implemented as a subclass of a standard numpy array
    """

    def __init__(
        self,
        value: ArrayLike,
        state_regs: Dict[int, str],
        _is_demodulated: bool = None,
        device: str = "cpu",
    ) -> None:
        """Initialize a SingleShot instance."""
        self._is_demodulated = False if _is_demodulated is None else _is_demodulated

        self.xp = ArrayModule(device)
        if not isinstance(value, self.xp.ndarray):
            value = self.xp.asarray(value)

        if not self.xp.issubdtype(value.dtype, self.xp.complexfloating):
            raise TypeError("value must be of `np.complexfloating` dtype")
        if value.ndim > 1 and value.shape[0] > 1 and self._is_demodulated is False:
            raise ValueError("value of complex dtype must be 1 dimensional")

        # if value.dtype != DEFAULT_DTYPE:
        #     value = value.astype(DEFAULT_DTYPE)

        if value.dtype != self.xp.dtype:
            value = value.astype(self.xp.dtype)

        self.value = value if value.ndim > 1 else value.reshape(1, -1)
        self._state_regs = state_regs

        self.id = str(uuid.uuid4())  # Generate a unique ID for the SingleShot instance
        self._plotter = ssp(children=self)

    def __getitem__(self, index) -> "SingleShot":
        """Get a slice of the value array.

        Args:
        index: Index or slice to retrieve.

        Returns:
        SingleShot: A new SingleShot instance with the sliced data.
        """
        regs_items = list(self.state_regs.items())
        if isinstance(index, tuple):
            new_reg_items = (
                [regs_items[index[0]]]
                if isinstance(index[0], int)
                else regs_items[index[0]]
            )
        else:
            new_reg_items = (
                [regs_items[index]] if isinstance(index, int) else regs_items[index]
            )

        new_instance = SingleShot(
            self.value[index],
            {item[0]: item[1] for item in new_reg_items},
            self._is_demodulated,
            self.device,
        )
        return new_instance

    def __repr__(self) -> str:
        """Return a string representation of the SingleShot instance.

        Returns:

            str: String representation of the SingleShot instance.
        """
        return f"SingleShot(value={self.value.get()}, state_regs='{self.state_regs}')"

    @property
    def is_demodulated(self) -> bool:
        """Indicates whether the SingleShot instance has been demodulated.

        Returns:
        bool: True if the data has been demodulated, False otherwise.
        """
        return self._is_demodulated

    @property
    def q_registers(self) -> str:
        """Get the qubit register string.

        Returns:
        str: A string representation of the qubit registers.
        """
        return "".join([str(q) for q in self.state_regs.keys()])

    @property
    def state(self) -> str:
        """Get the state string.

        Returns:
        str: A string representation of the qubit states.
        """
        return "".join(self.state_regs.values())

    @property
    def state_regs(self) -> Dict[int, str]:
        """Get the state registers dictionary.

        Returns:
        Dict[int, str]: A dictionary mapping qubit numbers to states.
        """
        return self._state_regs.copy()

    @property
    def shape(self) -> tuple:
        """Get the shape of the value array.

        Returns:
        tuple: Shape of the value array.
        """
        return self.value.shape

    @property
    def device(self):
        return self.xp.device

    def scatter(self, ax=None, **kwargs):
        # TODO: add docstring
        return self._plotter.scatter(ax, **kwargs)

    def plot(self, ax=None, x=None, **kwargs):
        # TODO: add docstring
        return self._plotter.plot(ax, x, **kwargs)

    def mean(self, axis: int = -1) -> "SingleShot":
        """Calculate the mean of the SingleShot values along the specified axis.

        This method computes the mean of the complex-valued data in the SingleShot
        instance. The mean is calculated element-wise for real and imaginary parts.

        Args:
            axis (int, optional): The axis along which to compute the mean.
                Defaults to -1 (last axis).

        Returns:
            SingleShot: A new SingleShot instance containing the mean values.

        Example:
            >>> import numpy as np
            >>> data = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])
            >>> state_regs = {0: '0', 1: '1'}
            >>> single_shot = SingleShot(data, state_regs)
            >>> mean_shot = single_shot.mean()
            >>> print(mean_shot)
            SingleShot(value=[[2.5+2.5j 3.5+3.5j 4.5+4.5j]], state_regs='{0: '0', 1: '1'}')
        """
        mean_value = _mean(self.value, axis)
        new_instance = SingleShot(
            mean_value, self.state_regs, self._is_demodulated, self.device
        )
        return new_instance

    def mean_filter(self, k):
        """Apply a mean filter to the SingleShot values.

        This method applies a mean filter (moving average) to the complex-valued
        data in the SingleShot instance. The filter is applied along the last axis
        of the data.

        Args:
            k (int): The size of the filter window. Must be an odd positive integer.

        Returns:
            SingleShot: A new SingleShot instance containing the filtered values.

        Raises:
            ValueError: If k is not positive integer.

        Example:
            >>> import numpy as np
            >>> data = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
            >>> state_regs = {0: '0', 1: '1'}
            >>> single_shot = SingleShot(data, state_regs)
            >>> filtered_shot = single_shot.mean_filter(3)
            >>> print(filtered_shot)
            SingleShot(value=[[1.5+1.5j 2. +2.j  3. +3.j  4. +4.j  4.5+4.5j]], state_regs='{0: '0', 1: '1'}')
        """
        if k <= 0:
            raise ValueError("k must be positive integer")
        new_value = _mean_filter(self.value, k, self.xp)
        new_instance = SingleShot(
            new_value, self.state_regs, self._is_demodulated, self.device
        )
        return new_instance

    def mean_centring(self, axis=-1) -> "SingleShot":
        """Center the SingleShot values by subtracting the mean."""
        centered_value = _mean_centring(self.value, axis)
        new_instance = SingleShot(
            centered_value, self.state_regs, self._is_demodulated, self.device
        )
        return new_instance

    def normalize(self, axis=-1) -> "SingleShot":
        # TODO: Add docstring
        norm_value = _normalize(self.value, axis)
        new_instance = SingleShot(
            norm_value, self.state_regs, self._is_demodulated, self.device
        )
        return new_instance

    def standardize(self, axis=-1) -> "SingleShot":
        """Standardize the SingleShot values by subtracting mean and dividing by standard deviation.

        Returns:
            SingleShot: A new SingleShot instance with standardized values.
        """
        standardized_value = _standardize(self.value, axis)
        new_instance = SingleShot(
            standardized_value, self.state_regs, self._is_demodulated, self.device
        )
        return new_instance

    def demodulate(
        self,
        intermediate_freq: Dict[int, float],
        meas_time: ArrayLike,
        direction: str = "clockwise",
    ) -> "SingleShot":
        # TODO: elaborate on docstring
        """Demodulate the SingleShot signal.

        Args:
            intermediate_freq (dict): Dictionary containing a qubit number as a key and an intermediate frequency of resonator as a value.
            meas_time (ArrayLike): Signal measurement time.
            direction (str): 'clockwise' for clockwise rotation, otherwise - else.

        Raises:
            ValueError: If the SingleShot is already demodulated.
            TypeError: If meas_time is not a 1D numpy array.

        Returns:
            SingleShot: A new SingleShot instance with demodulated values.
        """
        if self._is_demodulated:
            raise ValueError(
                "Cannot demodulate SingleShot which is already demodulated"
            )
        if not set(intermediate_freq.keys()).issubset(self.state_regs.keys()):
            raise ValueError(
                f"`intermediate_freq.keys()` must be a subset of `self.state_regs.keys()`, \
                    but got {intermediate_freq.keys()} and {self.state_regs.keys()}"
            )

        if not isinstance(meas_time, self.xp.ndarray):
            meas_time = self.xp.asarray(meas_time)
        if meas_time.ndim != 1:
            raise TypeError("meas_time must be a 1D array")

        if not self.shape[-1] == meas_time.shape[-1]:
            raise ValueError(
                f"Expecting `self` and `meas_time` have the same last dimension, but got {self.shape[-1]} and {meas_time.shape[-1]}"
            )

        value_new = _demodulate(
            self.value, intermediate_freq, meas_time, direction, self.xp
        )
        new_instance = SingleShot(value_new, self.state_regs, True, self.device)
        return new_instance

    def get_fft_amps_freqs(self, sampling_rate):
        # TODO: add docstring
        _, signal_length = self.shape
        freqs = self.xp.fft.fftfreq(signal_length, d=1.0 / sampling_rate)
        fft_results = self.xp.fft.fft(self.value, axis=1)
        amplitudes = self.xp.abs(fft_results) / signal_length
        return amplitudes, freqs

    def save(
        self,
        parent_dir: str,
        subfolder: str,
        verbose: bool = False,
    ) -> None:
        """Save the SingleShot instance to a specified directory.

        Args:
            parent_dir (str): The parent directory where the data will be saved.
            qubits_dir (str): The folder indicating qubit number set of the readout signal ('q1-q2-q3 for instance').
            subfolder (str): Subfolder within the state folder ('train', 'val', or 'test').
            verbose (bool): _description_
        """
        directory = os.path.join(parent_dir, self.q_registers, self.state, subfolder)
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.id}.npy")
        self.xp.save(file_path, self.value)
        if verbose:
            print(f"Saved {self.state} {subfolder} data to {file_path}")

    @classmethod
    def load(
        cls,
        parent_dir: str,
        qubits_dir: str,
        state: str,
        subfolder: str,
        index: int,
        verbose: bool = False,
    ) -> "SingleShot":
        """Load a SingleShot instance from a specified directory.

        Args:
            parent_dir (str): The parent directory where the data is stored.
            qubits_dir (str): The folder indicating qubit number set of the readout signal ('q1-q2-q3 for instance').
            state (str): _description_
            subfolder (str): Subfolder within the state folder ('train', 'val', or 'test').
            index (int): The index of the file to be loaded.
            verbose (bool): _description_

        Returns:
            SingleShot: The loaded SingleShot instance.
        """

        directory = os.path.join(parent_dir, qubits_dir, state, subfolder, "*.npy")
        dir_generator = glob.iglob(directory)
        filename = next(x for i, x in enumerate(dir_generator) if i == index)
        _id = os.path.splitext(os.path.basename(filename))[0]
        loaded_file = np.load(filename)
        if loaded_file.dtype == np.complex64:
            value = loaded_file
            is_demodulated = loaded_file.shape[0] > 1
        else:
            raise ValueError(
                "Unsupported dtype in loaded file. Must be 'np.complex64'."
            )
        state_regs = {int(q): s for q, s in zip(qubits_dir, state)}
        loaded_instance = cls(value, state_regs)
        loaded_instance.id = _id
        loaded_instance._is_demodulated = is_demodulated
        if verbose:
            print(f"[INFO] {filename} has been loaded.")
        return loaded_instance

    def to(self, device: str):
        if device == self.device:
            return self
        self.xp = ArrayModule(device)

        if isinstance(self.value, self.xp.ndarray) and device.startswith("cuda"):
            self.value = self.xp.array(self.value)
        elif hasattr(self.value, "get") and device == "cpu":  # CuPy array to NumPy
            self.value = self.value.get()
        else:
            self.value = self.xp.array(self.value)

        return self
