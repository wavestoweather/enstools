import numpy as np
import xarray
from skimage.metrics import structural_similarity

from enstools.core.errors import EnstoolsError

horizontal_spatial_dimensions = [("lat", "latitude"), ("lon", "longitude")]


def check_dimensions(data_array: xarray.DataArray):
    """
    Check that the provided data has at least two dimensions other than 'time'.
    """
    non_time_dimensions = [dim for dim in data_array.dims if dim != "time"]
    if len(non_time_dimensions) < 2:
        raise EnstoolsError("Data must have at least two dimensions other than 'time'.")




def structural_similarity_index(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:
    check_dimensions(reference)
    check_dimensions(target)

    # Load data
    reference.load()
    target.load()

    # Determine non-time dimensions and sort them by size, largest first
    non_time_dims = sorted([dim for dim in reference.dims if dim != 'time'],
                           key=lambda d: reference[d].size, reverse=True)

    if len(non_time_dims) < 2:
        raise ValueError("Data must have at least two spatial dimensions")

    # Use the two largest dimensions for SSIM computation
    largest_dims = non_time_dims[:2]

    # Compute SSIM for each 2D slice along the largest dimensions
    ref_slice = reference.isel({dim: 0 for dim in reference.dims if dim not in largest_dims})
    target_slice = target.isel({dim: 0 for dim in target.dims if dim not in largest_dims})

    ssim_value = compute_ssim_slice(ref_slice.values, target_slice.values)

    # Create a DataArray to store the SSIM value
    # Check if 'time' dimension exists in the data arrays
    if 'time' in reference.dims:
        result = xarray.DataArray(ssim_value, coords={'time': reference['time']}, dims=['time'])
    else:
        # If there's no 'time' dimension, create a DataArray without it
        result = xarray.DataArray(ssim_value)

    return result

    return result


def compute_ssim_slice(target: np.ndarray, reference: np.ndarray) -> float:
    """
    Returns the SSIM of a data slice. It uses the structural_similarity function from skimage.metrics.
    """
    ref_min, ref_max = np.min(reference), np.max(reference)
    target_min, target_max = np.min(target), np.max(target)

    true_min = min(ref_min, target_min)
    true_max = min(ref_max, target_max)

    return structural_similarity(target, reference, data_range=true_max - true_min)


def structural_similarity_log_index(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:
    ssim = structural_similarity_index(reference, target)

    return xarray.where(ssim >= 1.0, np.inf, -np.log10(1 - ssim))
