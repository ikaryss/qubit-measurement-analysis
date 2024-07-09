"Data processing functionality for collection of Shots"
import os
import glob
import warnings
from typing import List, Union

import numpy as np

from qubit_measurement_analysis.data.single_shot import SingleShot
from qubit_measurement_analysis.visualization.shot_collection_plotter import (
    CollectionPlotter as cp,
)


class ShotCollection:

    def __init__(self, singleshots: list[SingleShot] = None) -> None:

        if singleshots is not None:
            self.singleshots = singleshots
            # If singleshots is not none we need to check if several conditions are met:
            # 1. all data in signleshots list are SingleShot objects
            # 2. all singleshots have the same q_registers
            # TODO: 3. all singleshots are demodulated or not demodulated

            # Condition 1
            if not all(isinstance(s, SingleShot) for s in singleshots):
                raise TypeError("All elements must be instances of SingleShot")
            # condition 2
            if isinstance(self.q_registers, list):
                raise ValueError("All elements must be from the same qubits")
        else:
            self.singleshots: List[SingleShot] = []
        self._plotter = cp(children=self)

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
            return ShotCollection(new_shots)
        elif isinstance(index, slice):
            return ShotCollection(self.singleshots[index])
        else:
            return self.singleshots[index]

    def __len__(self) -> int:
        return len(self.singleshots)

    def __repr__(self) -> str:
        return f"ShotCollection with {len(self.singleshots)} SingleShot instances"

    @property
    def shape(self):
        # TODO: add description to the method
        return self.all_values.shape

    @property
    def is_demodulated(self):
        # TODO: Change method. I want to get the one unique is_demodulated flag across all singleshot in collection (True or False)
        return self.singleshots[0].is_demodulated

    def scatter(self, ax=None, **kwargs):
        # TODO: add docstring
        return self._plotter.scatter(ax, **kwargs)

    def append(self, object_: Union["SingleShot", "ShotCollection"]):
        """Append a SingleShot instance to the collection.

        Args:
            singleshots (Union[SingleShot, ShotCollection]): _description_
        """
        # TODO: need to check if object is aligned with data in class.
        # If self.singleshots is not empty we need to check if several conditions of object are met:
        # 1. all appended data in signleshots list are SingleShot objects
        # 2. all appended singleshots have the same q_registers
        # 3. appended all singleshots are demodulated or not demodulated

        # else if self.singleshits is empty list, append anyway

        if isinstance(object_, SingleShot):
            self.singleshots.append(object_)

        elif isinstance(object_, ShotCollection):
            self.singleshots.extend(object_.singleshots)
        else:
            raise TypeError(
                "Only SingleShot and ShotCollection instances can be appended"
            )

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
            axis = axis if axis == -1 else axis - 1
            return ShotCollection([shot.mean(axis) for shot in self.singleshots])

    @property
    def all_values(self):
        """Return a stacked array of all values."""
        values = [shot.value for shot in self.singleshots]
        return np.stack(values)

    @property
    def all_states(self):
        """Return a stacked array of all states."""
        states = [shot.state_regs for shot in self.singleshots]
        return np.stack(states)

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
        return ShotCollection(filtered_singleshots)

    def demodulate_all(
        self, intermediate_freq: dict, meas_time: np.ndarray, direction: str
    ) -> "ShotCollection":
        # TODO: add description to the function
        return ShotCollection(
            [
                s.demodulate(intermediate_freq, meas_time, direction)
                for s in self.singleshots
            ]
        )

    def mean_centring_all(self) -> "ShotCollection":
        # TODO: add description to the function
        return ShotCollection([s.mean_centring() for s in self.singleshots])

    def save_all(
        self,
        parent_dir: str,
        qubits_dir: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        clear_existing: bool = False,
        dtype: str = "complex64",
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
            dtype (str): Data type for saving ('complex64' or 'float32'). Defaults to 'complex64'.
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

        np.random.shuffle(self.singleshots)
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
            shot.save(parent_dir, qubits_dir, subfolder, dtype, verbose)

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
