import numpy as np
import xarray
from skimage.metrics import structural_similarity

from enstools.core.errors import EnstoolsError

horizontal_spatial_dimensions = ["lat", "lon"]


def check_coordinates(data_array: xarray.DataArray):
    """
    Check that the provided data has the horizontal spatial coordinates.
    """
    missing_dimensions = [d for d in horizontal_spatial_dimensions if d not in data_array.dims]
    if missing_dimensions:
        message = f"Structural similarity index expects data with dimensions {horizontal_spatial_dimensions}.\n" \
                  f"Dimension/s {missing_dimensions} missing."
        raise EnstoolsError(message)


def structural_similarity_index(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:
    r"""
    Returns the Structural Similarity Index Metric of the full DataArray.
    For more than two spatial dimensions the computation its done slice by slice.

    Relies on
    `skimage <https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity>`_
    to compute the SSIM of each slice.

    Parameters
    ----------
    reference : xarray.DataArray
    target : xarray.DataArray

    Returns
    -------
    SSIM: xarray.DataArray
        A data array with the time-series of the SSIM

    """
    check_coordinates(reference)

    # Load data
    reference.load()
    target.load()

    # Apply wrapper to horizontal slices
    res = xarray.apply_ufunc(compute_ssim_slice, reference, target,
                             #
                             vectorize=True,
                             input_core_dims=[horizontal_spatial_dimensions, horizontal_spatial_dimensions],
                             output_core_dims=[[]],
                             )
    pending_dimensions = [d for d in res.dims if d != "time"]

    if pending_dimensions:
        res = res.mean(dim=pending_dimensions)
    return res


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
