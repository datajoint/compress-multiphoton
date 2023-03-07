import numpy as np
from sklearn.linear_model import TheilSenRegressor


def _longest_run(bool_array):
    """Find the longest contiguous segment of True values inside bool_array.

    Args:
        bool_array: 1d boolean array.

    Returns:
        Slice with start and stop for the longest contiguous block of True values.
    """
    step = np.diff(np.int8(bool_array), prepend=0, append=0)
    on = np.where(step == 1)[0]
    off = np.where(step == -1)[0]
    i = np.argmax(off - on)
    return slice(on[i], off[i])


def compute_quantal_size(movie: np.array) -> dict:
    """Calculate quantal size for a movie.

    Args:
        movie (np.array):  A movie in the format (height, width, time).

    Returns:
        dict: A dictionary with the following keys:
            - 'model': The fitted TheilSenRegressor model.
            - 'min_intensity': Minimum intensity used.
            - 'max_intensity': Maximum intensity used.
            - 'variance': Variances at intensity levels.
            - 'quantal_size': Estimated quantal size.
            - 'zero_level': DC offset.
    """
    assert (
        movie.ndim == 3
    ), f"A three dimensional (Height, Width, Time) grayscale movie is expected, got {movie.ndim}"

    movie = movie.astype(np.int32, copy=False)
    intensity = (movie[:, :, :-1] + movie[:, :, 1:] + 1) // 2
    difference = movie[:, :, :-1] - movie[:, :, 1:]

    MIN_COUNTS = 100
    counts = np.bincount(intensity.flatten())
    counts_slice = _longest_run(counts > MIN_COUNTS)
    counts_slice = slice(
        max(counts_slice.stop * 20 // 100, counts_slice.start), counts_slice.stop
    )
    assert (
        counts_slice.stop - counts_slice.start > 0.10 * movie.max() 
    ), f"The image does not have a sufficient range of intensities to compute the noise transfer function."

    counts = counts[counts_slice]
    idx = (intensity >= counts_slice.start) & (intensity < counts_slice.stop)
    variance = (
        np.bincount(
            intensity[idx] - counts_slice.start,
            weights=(np.float32(difference[idx]) ** 2) / 2,
        )
        / counts
    )

    intensity_levels = np.r_[counts_slice]

    model = TheilSenRegressor()
    model.fit(intensity_levels.reshape(-1, 1), variance)
    quantal_size = model.coef_[0]
    zero_level = -model.intercept_ / model.coef_[0]

    return dict(
        model=model,
        min_intensity=counts_slice.start,
        max_intensity=counts_slice.stop - 1,
        variance=variance,
        quantal_size=quantal_size,
        zero_level=zero_level,
    )


def anscombe(frames, a0: float, a1: float, beta: float):
    """Compute the Anscombe variance stabilizing transform.

    Transforms a Poisson distributed signals in video recordings to...

    Args:
        frames (np.array_like): Single channel (gray scale) imaging frames, volume or video.
        a0 (float): Intercept of the photon transfer curve (offset)
        a1 (float): Slope of the photon transfer curve (ADC gain)
        beta (float): Ratio of the quantization step to noise.

    Returns:
        transformed_frames: Transformed frame.
    """
    transformed_frames = (
        2.0 / beta * np.sqrt(np.maximum(0, (frames + a0) / a1 + 0.375))
    ).astype(np.uint8)
    return transformed_frames
