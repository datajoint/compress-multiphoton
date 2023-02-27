import numpy as np
from sklearn.linear_model import TheilSenRegressor


def compute_quantal_size(scan):
    # Set some params
    num_frames = scan.shape[2]
    # min_count = num_frames * 0.1  # pixel values with fewer appearances will be ignored
    min_count = scan.min()
    max_acceptable_intensity = 3000  # pixel values higher than this will be ignored
    # max_acceptable_intensity = 100000

    # Make sure field is at least 32 bytes (int16 overflows if summed to itself)
    scan = scan.astype(np.float32, copy=False)

    # Create pixel values at each position in field
    eps = 1e-4  # needed for np.round to not be biased towards even numbers (0.5 -> 1, 1.5 -> 2, 2.5 -> 3, etc.)
    pixels = np.round((scan[:, :, :-1] + scan[:, :, 1:]) / 2 + eps)

    pixels = pixels.astype(np.int16 if np.max(abs(pixels)) < 2**15 else np.int32)

    # Compute a good range of pixel values (common, not too bright values)
    unique_pixels, counts = np.unique(pixels, return_counts=True)
    min_intensity = min(unique_pixels[counts > min_count])
    max_intensity = max(unique_pixels[counts > min_count])
    max_acceptable_intensity = min(max_intensity, max_acceptable_intensity)
    pixels_mask = np.logical_and(pixels >= min_intensity, pixels <= max_acceptable_intensity)

    # Select pixels in good range
    pixels = pixels[pixels_mask]
    unique_pixels, counts = np.unique(pixels, return_counts=True)

    # Compute noise variance
    variances = ((scan[:, :, :-1] - scan[:, :, 1:]) ** 2 / 2)[pixels_mask]
    pixels -= min_intensity
    variance_sum = np.zeros(len(unique_pixels))  # sum of variances per pixel value
    for i in range(0, len(pixels), int(1e8)):  # chunk it for memory efficiency
        variance_sum += np.bincount(
            pixels[i : i + int(1e8)], weights=variances[i : i + int(1e8)], minlength=np.ptp(unique_pixels) + 1
        )[unique_pixels - min_intensity]
    unique_variances = variance_sum / counts  # average variance per intensity

    # Compute quantal size (by fitting a linear regressor to predict the variance from intensity)
    X = unique_pixels.reshape(-1, 1)
    y = unique_variances
    model = TheilSenRegressor()  # robust regression
    model.fit(X, y)
    quantal_size = model.coef_[0]
    zero_level = -model.intercept_ / model.coef_[0]

    return (model, min_intensity, max_intensity, unique_pixels, unique_variances, quantal_size, zero_level)


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
    transformed_frames = (2.0 / beta * np.sqrt((frames + a0) / a1 + 0.375)).astype(np.int8)
    return transformed_frames


def compute_quantal_size(scan, max_acceptable_intensity=3000):
    # Set some params
    num_frames = scan.shape[2]
    min_count = num_frames * 0.1  # pixel values with fewer appearances will be ignored
    # min_count = scan.min()

    # Make sure field is at least 32 bytes (int16 overflows if summed to itself)
    scan = scan.astype(np.float32, copy=False)

    # Create pixel values at each position in field
    eps = 1e-4  # needed for np.round to not be biased towards even numbers (0.5 -> 1, 1.5 -> 2, 2.5 -> 3, etc.)
    pixels = np.round((scan[:, :, :-1] + scan[:, :, 1:]) / 2 + eps)

    pixels = pixels.astype(np.int16 if np.max(abs(pixels)) < 2**15 else np.int32)

    # Compute a good range of pixel values (common, not too bright values)
    unique_pixels, counts = np.unique(pixels, return_counts=True)
    min_intensity = min(unique_pixels[counts > min_count])
    max_intensity = max(unique_pixels[counts > min_count])
    max_acceptable_intensity = min(max_intensity, max_acceptable_intensity)
    pixels_mask = np.logical_and(pixels >= min_intensity, pixels <= max_acceptable_intensity)

    # Select pixels in good range
    pixels = pixels[pixels_mask]
    unique_pixels, counts = np.unique(pixels, return_counts=True)

    # Compute noise variance
    variances = ((scan[:, :, :-1] - scan[:, :, 1:]) ** 2 / 2)[pixels_mask]
    pixels -= min_intensity
    variance_sum = np.zeros(len(unique_pixels))  # sum of variances per pixel value
    for i in range(0, len(pixels), int(1e8)):  # chunk it for memory efficiency
        variance_sum += np.bincount(
            pixels[i : i + int(1e8)], weights=variances[i : i + int(1e8)], minlength=np.ptp(unique_pixels) + 1
        )[unique_pixels - min_intensity]
    unique_variances = variance_sum / counts  # average variance per intensity

    # Compute quantal size (by fitting a linear regressor to predict the variance from intensity)
    X = unique_pixels.reshape(-1, 1)
    y = unique_variances
    model = TheilSenRegressor()  # robust regression
    model.fit(X, y)
    quantal_size = model.coef_[0]
    zero_level = -model.intercept_ / model.coef_[0]

    output = dict(
        model=model,
        min_intensity=min_intensity,
        unique_pixels=unique_pixels,
        unique_variances=unique_variances,
        quantal_size=quantal_size,
        zero_level=zero_level,
    )

    return output
