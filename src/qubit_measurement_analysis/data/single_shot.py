"""Data processing functionality for Single-Shots.

This module provides the SingleShot class for storing and processing single-shot
measurement data in quantum experiments.
"""

import os
import uuid
import glob
from typing import Dict, Iterable

import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse

from qubit_measurement_analysis.visualization.single_shot_plotter import (
    SingleShotPlotter as ssp,
)

DEFAULT_DTYPE: type = np.complex64


class SingleShot:
    """A class for storing and processing a single SingleShot entity.

    This class represents a single-shot measurement in quantum experiments,
    providing methods for data manipulation, analysis, and visualization.

    This is implemented as a subclass of a standard numpy array
    """

    __last_is_demodulated = None

    def __init__(self, value: ArrayLike, state_regs: Dict[int, str]) -> None:
        """Initialize a SingleShot instance.

        Args:
        value (np.ndarray): A numpy array of `np.complex64` or `float` elements.
        state_regs (Dict[int, str]): A dictionary mapping qubit numbers to states.

        Raises:
        TypeError: If value is not a numpy array of complex or float elements.

        Example:
        >>> data = np.array([1+1j, 2+2j, 3+3j])
        >>> state_regs = {0: '0', 1: '1'}
        >>> single_shot = SingleShot(data, state_regs)
        """
        if not isinstance(value, Iterable):
            raise TypeError(
                "value must be an iterable of complex or float (int) elements"
            )

        if (
            value.ndim > 1
            and np.issubdtype(value.dtype, np.complexfloating)
            and SingleShot.__last_is_demodulated is None
        ):
            raise ValueError("value of complex dtype must be 1 dimensional")
        if not np.issubdtype(value.dtype, np.complexfloating):
            value = self._from_real(value)
        if value.dtype != DEFAULT_DTYPE:
            value = value.astype(DEFAULT_DTYPE)

        self.value = value if value.ndim > 1 else value.reshape(1, -1)
        self._state_regs = state_regs

        if SingleShot.__last_is_demodulated is not None:
            self._is_demodulated = SingleShot.__last_is_demodulated
        else:
            self._is_demodulated: bool = False

        self.id = str(uuid.uuid4())  # Generate a unique ID for the SingleShot instance
        self._plotter = ssp(children=self)

    def __getitem__(self, index) -> "SingleShot":
        """Get a slice of the value array.

        Args:
        index: Index or slice to retrieve.

        Returns:
        SingleShot: A new SingleShot instance with the sliced data.

        Example:
        >>> data = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j, 6+6j])
        >>> state_regs = {0: '0', 1: '1'}
        >>> single_shot = SingleShot(data, state_regs)
        >>> sliced_shot = single_shot[:, :3]
        >>> print(sliced_shot)
        SingleShot(value=[ 1.+1.j 2.+2.j 3.+3.j], state_regs='{0: '0', 1: '1'}')
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

        return SingleShot(
            self.value[index], {item[0]: item[1] for item in new_reg_items}
        )

    def __repr__(self) -> str:
        """Return a string representation of the SingleShot instance.

        Returns:

            str: String representation of the SingleShot instance.
        """
        return f"SingleShot(value={np.array2string(self.value, threshold=5)}, state_regs='{self.state_regs}')"

    @property
    def is_demodulated(self) -> bool:
        """Indicates whether the SingleShot instance has been demodulated.

        Returns:
        bool: True if the data has been demodulated, False otherwise.

        Example:
        >>> data = np.array([1+1j, 2+2j, 3+3j])
        >>> state_regs = {0: '0', 1: '1'}
        >>> single_shot = SingleShot(data, state_regs)
        >>> print(single_shot.is_demodulated)
        False
        """
        return self._is_demodulated

    @property
    def q_registers(self) -> str:
        """Get the qubit register string.

        Returns:
        str: A string representation of the qubit registers.

        Example:
        >>> data = np.array([1+1j, 2+2j, 3+3j])
        >>> state_regs = {0: '0', 1: '1'}
        >>> single_shot = SingleShot(data, state_regs)
        >>> print(single_shot.q_registers)
        '01'
        """
        return "".join([str(q) for q in self.state_regs.keys()])

    @property
    def state(self) -> str:
        """Get the state string.

        Returns:
        str: A string representation of the qubit states.

        Example:
        >>> data = np.array([1+1j, 2+2j, 3+3j])
        >>> state_regs = {0: '0', 1: '1'}
        >>> single_shot = SingleShot(data, state_regs)
        >>> print(single_shot.state)
        '01'
        """
        return "".join(self.state_regs.values())

    @property
    def state_regs(self) -> Dict[int, str]:
        """Get the state registers dictionary.

        Returns:
        Dict[int, str]: A dictionary mapping qubit numbers to states.

        Example:
        >>> data = np.array([1+1j, 2+2j, 3+3j])
        >>> state_regs = {0: '0', 1: '1'}
        >>> single_shot = SingleShot(data, state_regs)
        >>> print(single_shot.state_regs)
        {0: '0', 1: '1'}
        """
        return self._state_regs.copy()

    @property
    def shape(self) -> tuple:
        """Get the shape of the value array.

        Returns:
        tuple: Shape of the value array.

        Example:
        >>> data = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j, 6+6j])
        >>> state_regs = {0: '0', 1: '1'}
        >>> single_shot = SingleShot(data, state_regs)
        >>> print(single_shot.shape)
        (1, 6)
        """
        return self.value.shape

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
        mean_value = self.value.mean(axis, keepdims=True)
        return SingleShot(mean_value, self.state_regs)

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
        p = self.shape[-1]
        diag_offset = np.linspace(-(k // 2), k // 2, k, dtype=int)
        sparse_matrix = scipy.sparse.diags(
            np.ones((k, p)), offsets=diag_offset, shape=(p, p)
        )
        nrmlize = np.ones_like(self.value) @ sparse_matrix
        new_value = ((self.value @ sparse_matrix) / nrmlize).astype(DEFAULT_DTYPE)
        return SingleShot(new_value, self.state_regs)

    def mean_centring(self) -> "SingleShot":
        """Center the SingleShot values by subtracting the mean.

        Returns:
        SingleShot: A new SingleShot instance with centered values.

        Example:
        >>> data = np.array([1+1j, 2+2j, 3+3j])
        >>> state_regs = {0: '0', 1: '1'}
        >>> single_shot = SingleShot(data, state_regs)
        >>> centered = single_shot.mean_centring()
        >>> print(centered)
        SingleShot(value=[-1.-1.j 0.+0.j 1.+1.j], state_regs='{0: '0', 1: '1'}')
        """
        centered_value = self.value - self.mean().value
        return SingleShot(centered_value, self.state_regs)

    # TODO: rewrite w.r.t complex values
    def normalize(self) -> "SingleShot":
        """Normalize the SingleShot values to have unit norm.

        Returns:
            SingleShot: A new SingleShot instance with normalized values.
        """
        norm_value = self.value / np.linalg.norm(self.value, axis=-1)
        return SingleShot(norm_value, self.state_regs)

    # TODO: rewrite w.r.t complex values
    def standardize(self) -> "SingleShot":
        """Standardize the SingleShot values by subtracting mean and dividing by standard deviation.

        Returns:
            SingleShot: A new SingleShot instance with standardized values.
        """
        standardized_value = (self.value - self.value.mean()) / self.value.std(-1)
        return SingleShot(standardized_value, self.state_regs)

    def get_fft_amps_freqs(self, sampling_rate):
        # TODO: add docstring
        _, signal_length = self.shape
        freqs = np.fft.fftfreq(signal_length, d=1.0 / sampling_rate)
        fft_results = np.fft.fft(self.value, axis=1)
        amplitudes = np.abs(fft_results) / signal_length
        return amplitudes, freqs

    def demodulate(
        self,
        intermediate_freq: Dict[int, float],
        meas_time: np.ndarray,
        direction: str = "clockwise",
    ) -> "SingleShot":
        # TODO: elaborate on docstring
        """Demodulate the SingleShot signal.

        Args:
            intermediate_freq (dict): Dictionary containing a qubit number as a key and an intermediate frequency of resonator as a value.
            meas_time (np.ndarray): Signal measurement time.
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
            # TODO: add value error description that keys from `intermediate_freq` and `self.state_regs`
            # must not to be differ
            raise ValueError("_description_")

        if not isinstance(meas_time, np.ndarray) or meas_time.ndim != 1:
            raise TypeError("meas_time must be a 1D numpy array")

        if not self.shape[-1] == meas_time.shape[-1]:
            raise ValueError(
                f"Expecting `self` and `meas_time` have the same last dimension, but got {self.shape[-1]} and {meas_time.shape[-1]}"
            )

        num_freqs = len(intermediate_freq)
        value_new = np.tile(self.value, (num_freqs, 1))
        meas_time_new = np.tile(meas_time, (num_freqs, 1))
        intermediate_freqs = np.array(list(intermediate_freq.values())).reshape(-1, 1)
        value_new = self._exponential_rotation(
            value_new, intermediate_freqs, meas_time_new, direction
        )
        SingleShot.__last_is_demodulated = True
        return SingleShot(value_new, self.state_regs)

    @staticmethod
    def _exponential_rotation(value, freqs, times, direction) -> np.ndarray:
        # TODO: elaborate on docstring
        """Apply exponential rotation to the value array.

        Args:
            value (np.ndarray): Input array.
            freqs (np.ndarray): Array of intermediate frequencies.
            times (np.ndarray): Array of measurement times.
            direction (str): 'clockwise' for clockwise rotation, otherwise - else.

        Returns:
            np.ndarray: Array after applying exponential rotation.
        """
        phase = 2 * np.pi * freqs * times
        rotation = (
            np.exp(-1j * phase) if direction == "clockwise" else np.exp(1j * phase)
        )
        return (value * rotation).astype(DEFAULT_DTYPE)

    def save(
        self,
        parent_dir: str,
        subfolder: str,
        dtype: str = "complex64",
        verbose: bool = False,
    ) -> None:
        """Save the SingleShot instance to a specified directory.

        Args:
            parent_dir (str): The parent directory where the data will be saved.
            qubits_dir (str): The folder indicating qubit number set of the readout signal ('q1-q2-q3 for instance').
            subfolder (str): Subfolder within the state folder ('train', 'val', or 'test').
            dtype (str): Data type for saving ('complex64' or 'float32'). Defaults to 'complex64'.
            verbose (bool): _description_
        """
        directory = os.path.join(parent_dir, self.q_registers, self.state, subfolder)
        os.makedirs(directory, exist_ok=True)
        save_value = self._prepare_save_value(dtype)
        file_path = os.path.join(directory, f"{self.id}.npy")
        np.save(file_path, save_value)
        if verbose:
            print(f"Saved {self.state} {subfolder} data to {file_path}")

    def _prepare_save_value(self, dtype: str) -> np.ndarray:
        """Prepare the value array for saving.

        Args:
            dtype (str): Data type for saving ('complex64' or 'float32').

        Returns:
            np.ndarray: Value array prepared for saving.
        """
        if dtype == "complex64":
            return self.value
        elif dtype == "float32":
            return self._to_float32()
        else:
            raise ValueError("Unsupported dtype. Use 'complex64' or 'float32'.")

    def _to_float32(self) -> np.ndarray:
        """Convert the value array to float32 dtype.

        Returns:
            np.ndarray: Value array converted to float32 dtype.
        """
        if self._is_demodulated:
            save_value = np.empty(
                (2 * self.value.shape[0], self.value.shape[1]), dtype=np.float32
            )
            save_value[0::2] = self.value.real
            save_value[1::2] = self.value.imag
        else:
            save_value = np.empty((2, self.value.shape[1]), dtype=np.float32)
            save_value[0] = self.value.real
            save_value[1] = self.value.imag
        return save_value

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
        if loaded_file.dtype == np.float32:
            value = cls._from_real(loaded_file)
            is_demodulated = loaded_file.shape[0] != 2
        elif loaded_file.dtype == DEFAULT_DTYPE:
            value = loaded_file
            is_demodulated = loaded_file.shape[0] > 1
        else:
            raise ValueError(
                "Unsupported dtype in loaded file. Must be 'float32' or 'complex64'."
            )
        state_regs = {int(q): s for q, s in zip(qubits_dir, state)}
        loaded_instance = cls(value, state_regs)
        loaded_instance.id = _id
        loaded_instance._is_demodulated = is_demodulated
        if verbose:
            print(f"[INFO] {filename} has been loaded.")
        return loaded_instance

    @staticmethod
    def _from_real(loaded_file: np.ndarray) -> np.ndarray:
        """Convert a float or int array to complex64.

        Args:
            loaded_file (np.ndarray): Input array of dtype float.

        Returns:
            np.ndarray: Complex64 array.
        """
        if loaded_file.shape[0] % 2 != 0:
            raise ValueError(
                "Invalid shape for float data. Must have even first dimension."
            )
        real_part = loaded_file[0::2]
        imag_part = loaded_file[1::2]
        value = real_part + 1j * imag_part
        return value.astype(DEFAULT_DTYPE)
