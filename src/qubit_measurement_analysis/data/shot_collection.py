"Data processing functionality for collection of Shots"
import os
import glob
import warnings
from functools import cached_property
from typing import List, Callable, Iterator, Iterable

from qubit_measurement_analysis.data.single_shot import SingleShot
from qubit_measurement_analysis._array_module import ArrayModule
from qubit_measurement_analysis.visualization.shot_collection_plotter import (
    CollectionPlotter as cp,
)
from qubit_measurement_analysis._transformations import (
    _mean,
    _mean_filter,
    _mean_centring,
    _normalize,
    _standardize,
    _demodulate,
)


class ShotCollection:
    # __slots__ = ("xp", "singleshots", "_plotter")

    def __init__(
        self, singleshots: list[SingleShot] = None, device: str = "cpu"
    ) -> None:
        self.xp = ArrayModule(device)
        self.singleshots = []
        self._is_demodulated = None
        self._plotter = cp(children=self)

        if singleshots:
            self.extend(singleshots)

    def _validate_shots(self, shots: List[SingleShot]) -> None:
        if not shots:
            return  # Empty list is valid

        # 1. Check if all data in shots list are SingleShot objects
        if not all(isinstance(shot, SingleShot) for shot in shots):
            raise TypeError("All elements in shots must be SingleShot objects")

        # 2. Check if all singleshots have the same state_regs.keys()
        first_shot_keys = set(shots[0].state_regs.keys())
        if not all(set(shot.state_regs.keys()) == first_shot_keys for shot in shots):
            raise ValueError(
                "All SingleShot objects must have the same state_regs keys"
            )

        # 3. Check if all singleshots are either all demodulated or all not demodulated
        first_shot_demodulated = shots[0].is_demodulated
        if not all(shot.is_demodulated == first_shot_demodulated for shot in shots):
            raise ValueError(
                "All SingleShot objects must have the same demodulation status"
            )
        self._is_demodulated = first_shot_demodulated

    def _apply_vectorized(self, func: Callable, **kwargs) -> "ShotCollection":
        processed_values = func(self.all_values, **kwargs)

        new_shots = [
            SingleShot(value, state_regs, self.is_demodulated, self.device)
            for value, state_regs in zip(processed_values, self.all_states)
        ]

        return ShotCollection(new_shots, self.device)

    def __getitem__(self, index) -> SingleShot:
        if isinstance(index, tuple):
            # The first element of the index is for selecting Shots
            shot_indices = index[0]
            # The remaining elements of the index are for each Shot's array
            shot_slices = index[1:]
            # Apply indices
            selected_shots = self.singleshots[shot_indices]
            # If selected_shots is a single Shot, make it a list
            if isinstance(selected_shots, SingleShot):
                selected_shots = [selected_shots]
            # Apply shot slices
            new_shots = [shot[shot_slices] for shot in selected_shots]
            return ShotCollection(new_shots, self.device)
        elif isinstance(index, slice):
            return ShotCollection(self.singleshots[index], self.device)
        else:
            return self.singleshots[index]

    def __len__(self) -> int:
        return len(self.singleshots)

    def __repr__(self) -> str:
        return f"ShotCollection(n_shots={len(self)}, device='{self.device}')"

    def __iter__(self) -> Iterator[SingleShot]:
        return iter(self.singleshots)

    @property
    def device(self) -> str:
        return self.xp.device

    def to(self, device: str) -> "ShotCollection":
        if device == self.device:
            return self
        self.xp = ArrayModule(device)
        for shot in self.singleshots:
            shot.to(device)
        return self

    @property
    def shape(self):
        # TODO: add description to the method
        return self.all_values.shape

    @property
    def is_demodulated(self):
        # TODO: Change method. I want to get the one unique is_demodulated flag across all singleshot in collection (True or False)
        return self._is_demodulated

    def scatter(self, ax=None, **kwargs):
        # TODO: add docstring
        return self._plotter.scatter(ax, **kwargs)

    def append(self, shot: SingleShot) -> None:
        self.extend([shot])

    def extend(self, shots: List[SingleShot]) -> None:
        self._validate_shots(shots)
        self.singleshots.extend(shots)

    def mean(self, axis: int = -1) -> "SingleShot":
        # TODO: add description to the function
        if len(self.singleshots) == 0:
            raise ValueError("SignalCollection is empty")

        if axis == 0:
            state_regs = self.singleshots[0].state_regs
            # check if all signleshots are of the same state
            if len(self.unique_states) != 1:
                state_regs = {reg: "<UNK>" for reg in list(self.q_registers)}
                warnings.warn(
                    "ShotCollection contains more than 1 unique state. Taking mean regardless of state. Assigned state is '<UNK>'"
                )
            return SingleShot(self.all_values.mean(axis), state_regs)
        else:
            return self._apply_vectorized(_mean, axis=axis)

    @property
    def all_values(self):
        """Return a stacked array of all values."""
        values = [shot.value for shot in self.singleshots]
        return self.xp.stack(values)

    @property
    def all_states(self):
        """Return a stacked array of all states."""
        states = [shot.state_regs for shot in self.singleshots]
        return states

    @property
    def unique_states(self):
        # TODO: add description to the method
        unique_states = set()  # Using a set to store unique classes
        for shot in self.singleshots:
            unique_states.add(shot.state)
        # Converting the set to a list before returning
        return list(unique_states)

    @property
    def q_registers(self):
        # TODO: add description to the method

        unique_registers = set()  # Using a set to store unique classes

        for shot in self.singleshots:

            unique_registers.add(shot.q_registers)
        return (
            list(unique_registers)
            if len(unique_registers) > 1
            else list(unique_registers)[0]
        )

    def filter_by_state(self, target) -> "ShotCollection":
        # TODO: add description to the function
        if isinstance(target, str):
            target = [target]
        target_set = set(target)
        filtered_singleshots = [
            shot for shot in self.singleshots if shot.state in target_set
        ]
        return ShotCollection(filtered_singleshots, self.device)

    def demodulate_all(
        self, intermediate_freq: dict, meas_time: Iterable, direction: str
    ) -> "ShotCollection":
        # TODO: add description to the function
        self._is_demodulated = True
        return self._apply_vectorized(
            _demodulate,
            intermediate_freq=intermediate_freq,
            meas_time=meas_time,
            direction=direction,
            module=self.xp,
        )

    def mean_centring_all(self, axis=-1) -> "ShotCollection":
        # TODO: add description to the function
        return self._apply_vectorized(_mean_centring, axis=axis)

    def mean_filter_all(self, k):
        return self._apply_vectorized(_mean_filter, k=k, module=self.xp)

    def normalize_all(self, axis=-1):
        return self._apply_vectorized(_normalize, axis=axis)

    def standardize_all(self, axis=-1):
        return self._apply_vectorized(_standardize, axis=axis)

    def save_all(
        self,
        parent_dir: str,
        qubits_dir: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        clear_existing: bool = False,
        verbose: bool = False,
    ) -> None:
        """Save the ShotCollection to the specified directory with train, val, test splits

        Args:
            parent_dir (str): The parent directory where the data will be saved.
            qubits_dir (str): The folder indicating qubit number set of the readout signal ('q1-q2-q3 for instance').
            train_ratio (float): _description_
            val_ratio (float): _description_
            test_ratio (float): _description_
            clear_existing (bool, optional): _description_. Defaults to False.
            verbose (bool): _description_
        """
        if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
            raise ValueError("Ratios must be between 0 and 1")
        if not abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6:
            raise ValueError("Ratios must sum up to 1")
        if clear_existing:
            print("[INFO] Deleting files...")
            for state in self.unique_states:
                for subfolder in ["train", "val", "test"]:
                    directory = os.path.join(parent_dir, qubits_dir, state, subfolder)
                    for file in glob.glob(os.path.join(directory, "*.npy")):
                        os.remove(file)

        self.xp.random.shuffle(self.singleshots)
        num_shots = len(self.singleshots)
        train_end = int(train_ratio * num_shots)
        val_end = train_end + int(val_ratio * num_shots)
        for idx, shot in enumerate(self.singleshots):
            if idx < train_end:
                subfolder = "train"
            elif idx < val_end:
                subfolder = "val"
            else:
                subfolder = "test"
            shot.save(parent_dir, qubits_dir, subfolder, verbose)

    @classmethod
    def load(
        cls,
        parent_dir: str,
        qubits_dir: str,
        state: str,
        subfolder: str,
        num_samples: int = None,
        verbose: bool = False,
    ) -> "ShotCollection":
        """Load a specified number of SingleShot instances from a directory.

        Args:
            parent_dir (str): The parent directory where the data will be saved.
            qubits_dir (str): The folder indicating qubit number set of the readout signal ('q1-q2-q3 for instance').
            state (str): The state of the system.
            subfolder (str): Subfolder within the state folder ('train', 'val', or 'test').
            num_samples (int, optional): _description_. Defaults to None.
            verbose (bool): _description_

        Returns:
            ShotCollection: _description_
        """
        collection = cls()
        for idx in range(num_samples):
            singleshot = SingleShot.load(
                parent_dir, qubits_dir, state, subfolder, idx, verbose
            )
            collection.append(singleshot)
        return collection
