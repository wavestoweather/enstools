import numpy as np
import xarray

from .normalized_root_mean_square_error import mean_square_error


def peak_signal_to_noise_ratio(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:
    """
    Description:
        It returns the Peak Sign to Noise Ratio.
        With images, it is usually defined as:
        20*log10(max_value)-10*log10(mse)
        with all the values being between 0 and max_value.
        However, for real values I found this definition:
        20*log10(range)-10*log10(mse)

    Functions
    :param reference:
    :param target:
    :return:
    """

    non_temporal_dimensions = [d for d in reference.dims if d != "time"]
    # Compute range from the reference file
    ref_min, ref_max = reference.min(dim=non_temporal_dimensions), reference.max(dim=non_temporal_dimensions)
    ref_range = ref_max - ref_min

    return 20 * np.log10(ref_range) - 10 * np.log10(mean_square_error(reference, target))
