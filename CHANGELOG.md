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
