## 0.6.1 (2024-08-09)

### Refactor

- **src/qubit_measurement_analysis/data/single_shot.py**: add __slots__ to SingleShot class
- change extend functionality in ShotCollection class for performance improvements, add small refractor changes
- add minor improvements

## 0.6.0 (2024-08-06)

### Feat

- **src/qubit_measurement_analysis/data/shot_collection.py**: add Lazy Evaluation to ShotCollection class
- move all data transformation functionality from SingleShot and ShotCollection to a single file _transformations.py
- add device handling to ShotCollection class
- **src/qubit_measurement_analysis/data/single_shot.py**: add gpu support to SingleShot class

### Refactor

- **src/qubit_measurement_analysis/data/single_shot.py**: minor code changes

## 0.5.1 (2024-07-26)

### Fix

- **src/qubit_measurement_analysis/data/single_shot.py**: fix normalize and standardize functions

## 0.5.0 (2024-07-24)

### Feat

- remove auto real -> complex conversion when instantiating SingleShot object, remove optional complex -> real conversion when saving a SingleShot instance

### Fix

- **src/qubit_measurement_analysis/data/single_shot.py**: fix a self.is_demodulated setter after initialization
- **src/qubit_measurement_analysis/data/single_shot.py**: return new SingleShot instance after mean_filter function
- **src/qubit_measurement_analysis/data/single_shot.py**: fix a bug when self.is_demodulated flag had not been remembered by class after data manupulation
- **src/qubit_measurement_analysis/data/single_shot.py**: make state_regs property read-only

### Refactor

- **src/qubit_measurement_analysis/data/single_shot.py**: change the `self._is_demodulated` initialization
- add minor code changes, prepare package for GPU support

## 0.4.1 (2024-07-23)

### Fix

- fix a github workflow actions bug

## 0.4.0 (2024-07-23)

### Feat

- extend visualization functionality by allowing the use of built-in matplotlib arguments in the .plt and .scatter functions

### Fix

- **SingleShotPlotter**: correct a typo in SingleShotPlotter class naming
- **SingleShot**: hot fix of previous commit
- **SingleShot**: fix a bug when creating SingleShot instance with array of complex but not `np.complex64` values

### Refactor

- **src/qubit_measurement_analysis/visualization/**: refractor code, reduce unnecessary dependencies
- **.gitignore**: add .vscode to .gitignore

## 0.3.0 (2024-07-09)

### Feat

- **src/qubit_measurement_analysis/data/shot_collection.py-src/qubit_measurement_analysis/visualization/shot_collection_plotter.py**: add functionality to control a collection of single-shots
- **src/qubit_measurement_analysis/visualization**: add basic visualization functionality

## 0.2.0 (2024-07-09)

### Feat

- **SingleShot**: add singleshot class for analyzing signal from single qubit measurement

### Refactor

- **src/qubit_measurement_analysis/data/single_shot.py**: raise an exception if shape mismatch in `_exponential_rotation` function
- Initial commit
