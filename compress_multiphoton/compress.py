import numpy as np
from sklearn.linear_model import HuberRegressor as Regressor
import imageio as io


def _longest_run(bool_array: np.ndarray) -> slice:
    """
    Find the longest contiguous segment of True values inside bool_array.
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


def compute_sensitivity(movie: np.array, count_weight_gamma: float=0.2) -> dict:
    """Calculate photon sensitivity

    Args:
        movie (np.array):  A movie in the format (height, width, time).
        count_weight_gamma: 0.00001=weigh each intensity level equally, 
            1.0=weigh each intensity in proportion to pixel counts.

    Returns:
        dict: A dictionary with the following keys:
            - 'model': The fitted TheilSenRegressor model.
            - 'min_intensity': Minimum intensity used.
            - 'max_intensity': Maximum intensity used.
            - 'variance': Variances at intensity levels.
            - 'sensitivity': Sensitivity.
            - 'zero_level': X-intercept.
    """
    assert (
        movie.ndim == 3
    ), f"A three dimensional (Height, Width, Time) grayscale movie is expected, got {movie.ndim}"

    movie = np.maximum(0, movie.astype(np.int32, copy=False))
    intensity = (movie[:, :, :-1] + movie[:, :, 1:] + 1) // 2
    difference = movie[:, :, :-1].astype(np.float32) - movie[:, :, 1:]

    select = intensity > 0
    intensity = intensity[select]
    difference = difference[select]

    counts = np.bincount(intensity.flatten())
    bins = _longest_run(counts > 0.01 * counts.mean())  # consider only bins with at least 1% of mean counts 
    bins = slice(max(bins.stop * 3 // 100, bins.start), bins.stop)
    assert (
        bins.stop - bins.start > 100
    ), f"The image does not have a sufficient range of intensities to compute the noise transfer function."

    counts = counts[bins]
    idx = (intensity >= bins.start) & (intensity < bins.stop)
    variance = (
        np.bincount(
            intensity[idx] - bins.start,
            weights=(difference[idx] ** 2) / 2,
        )
        / counts
    )
    model = Regressor()
    model.fit(np.c_[bins], variance, counts ** count_weight_gamma)
    sensitivity = model.coef_[0]
    zero_level = - model.intercept_ / model.coef_[0]

    return dict(
        model=model,
        counts=counts,
        min_intensity=bins.start,
        max_intensity=bins.stop,
        variance=variance,
        sensitivity=sensitivity,
        zero_level=zero_level,
    )

def make_luts(zero_level: int, sensitivity: float, input_max: int, beta: float=0.5):
	"""
	Compute lookup tables LUT1 and LUT2.
	LUT1 converts a linear grayscale image into a uniform variance image. 
	LUT2 is the inverse of LUT1 
	:param zero_level: the input level correspoding to zero photon rate
	:param sensitivity: the size of one photon in the linear input image.
	:param beta: the grayscale quantization step in units of noise std dev
	"""
	# produce anscombe LUT1 and LUT2
	xx = (np.r_[:input_max + 1] - zero_level) / sensitivity
	zero_slope = 1 / beta / np.sqrt(3/8)
	offset = -xx.min() * zero_slope
	LUT1 = np.uint8(
		offset +
		(xx < 0) * (xx * zero_slope) +
		(xx >= 0) * (2.0 / beta * (np.sqrt(np.maximum(0, xx) + 3/8) - np.sqrt(3/8))))
	_, LUT2 = np.unique(LUT1, return_index=True)
	LUT2 += (np.r_[:LUT2.size] / LUT2.size * (LUT2[-1] - LUT2[-2])/2).astype('int16')
	return LUT1, LUT2.astype('int16')


def lookup(movie, LUT):
    """
    Apply lookup table LUT to input movie
    """
    return LUT[np.maximum(0, np.minimum(movie, LUT.size-1))]


def save_movie(movie, path, scale, format='gif'):
    if format == "gif":
        with io.get_writer(path, mode='I', duration=.01, loop=False) as f:
            for frame in movie:
                f.append_data(scale * frame)
    else:
        raise NotImplementedError(f"Format {format} is not implemented")
