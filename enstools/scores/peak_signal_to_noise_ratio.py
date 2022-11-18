import numpy as np
import xarray

from .normalized_root_mean_square_error import mean_square_error


def peak_signal_to_noise_ratio(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:
    r"""
     It returns the Peak Sign to Noise Ratio.
        With images, it is usually defined as:

        .. math::
            20 \cdot log10(maxValue)-10 \cdot log10(mse)

        with all the values being between 0 and max_value.

        However, because we are dealing with real values we will use:

        .. math::
            20 \cdot log10(range)-10 \cdot log10(mse)

    Parameters
    ----------
    reference : xarray.DataArray
    target : xarray.DataArray

    Returns
    -------
    psnr: xarray.DataArray
        A data array with the time-series of the peak signal to noise ratio.

    """

    non_temporal_dimensions = [d for d in reference.dims if d != "time"]
    # Compute range from the reference file
    ref_min, ref_max = reference.min(dim=non_temporal_dimensions), reference.max(dim=non_temporal_dimensions)
    ref_range = ref_max - ref_min

    return 20 * np.log10(ref_range) - 10 * np.log10(mean_square_error(reference, target))
