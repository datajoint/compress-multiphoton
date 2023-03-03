import numpy as np
from sklearn.linear_model import TheilSenRegressor


def _longest_run(bool_array):
    step = np.diff(np.int8(bool_array), prepend=0, append=0)
    on = np.where(step == 1)[0]
    off = np.where(step == -1)[0]
    i = np.argmax(off - on)
    return slice(on[i], off[i])


def compute_quantal_size(movie):
    movie = movie.astype(np.int32, copy=False)
    intensity = (movie[:, :, :-1] + movie[:, :, 1:] + 1) // 2
    difference = movie[:, :, :-1] - movie[:, :, 1:]

    CTS_THRESHOLD = 1000
    cts = np.bincount(intensity.flatten())
    cts_slice = _longest_run(cts > CTS_THRESHOLD)
    cts_slice = slice(max(cts_slice.stop * 20 // 100, cts_slice.start), cts_slice.stop)

    counts = counts[cts_slice]
    idx = (intensity >= cts_slice.start) & (intensity < cts_slice.stop)
    variance = (
        np.bincount(
            intensity[idx] - cts_slice.start,
            weights=(np.float32(difference[idx]) ** 2) / 2,
        )
        / counts
    )

    intensity_levels = np.arange(cts_slice.start, cts_slice.stop)

    model = TheilSenRegressor()
    model.fit(intensity_levels.reshape(-1, 1), variance)
    quantal_size = model.coef_[0]
    zero_level = -model.intercept_ / model.coef_[0]

    output = dict(
        model=model,
        min_intensity=cts_slice.start,
        max_intensity=cts_slice.stop,
        unique_intensities=intensity_levels,
        unique_variances=variance,
        quantal_size=quantal_size,
        zero_level=zero_level,
    )

    return output


def anscombe(frames, a0: float, a1: float, beta: float):
    """Compute the Anscombe variance stabilizing transform.

    Transforms a Poisson distributed signals in video recordings to...

    Args:
        frames (np.array_like): Single channel (gray scale) imaging frames, volume or video.
        a0 (float): Intercept of the photon transfer curve (offset)
        a1 (float): Slope of the photon transfer curve (ADC gain)
        beta (float): Ratio of the quantization step to noise.

    Returns:
        transformed_frames: _description_
    """
    transformed_frames = (2.0 / beta * np.sqrt((frames + a0) / a1 + 0.375)).astype(
        np.int8
    )
    return transformed_frames
