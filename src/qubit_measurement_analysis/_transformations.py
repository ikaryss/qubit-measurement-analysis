def _mean(array, axis):
    return array.mean(axis, keepdims=True)


def _mean_filter(array, k, module):
    p = array.shape[-1]
    diag_offset = module.np.linspace(-(k // 2), k // 2, k, dtype=int)
    sparse_matrix = module.scipy.sparse.diags(
        module.np.ones((k, p)), offsets=diag_offset, shape=(p, p)
    ).toarray()
    nrmlize = module.np.ones_like(array) @ sparse_matrix
    new_array = (array @ sparse_matrix) / nrmlize
    return new_array


def _mean_centring(array, axis=-1):
    return array - array.mean(axis=axis, keepdims=True)


def _normalize(array, axis=-1):
    min_array = array.min(axis=axis, keepdims=True)
    max_array = array.max(axis=axis, keepdims=True)
    normalized = (array - min_array) / (max_array - min_array)
    return normalized


def _standardize(array, axis=-1):
    std_real = array.real.std(axis=axis, keepdims=True)
    std_imag = array.imag.std(axis=axis, keepdims=True)
    standardized = (array - array.mean(axis=axis, keepdims=True)) / (
        std_real + std_imag * 1j
    )
    return standardized


def _demodulate(array, intermediate_freq, meas_time, direction, module):
    # Conversion from dict to array
    intermediate_freq = module.array(list(intermediate_freq.values())).reshape(-1, 1)
    # Reshape meas_time to match the shape required for broadcasting
    meas_time = meas_time.reshape(1, -1)
    # Calculate phase using broadcasting
    phase = 2 * module.pi * intermediate_freq @ meas_time
    # Calculate rotation using broadcasting
    rotation = (
        module.exp(-1j * phase) if direction == "clockwise" else module.exp(1j * phase)
    )
    # Perform the rotation on the array
    demodulated = array * rotation
    return demodulated
