import numpy as np
import xarray


def mean_square_error(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:
    r"""
    Compute the mean square error:

    .. math::
        \frac{1}{n}\sum_{i=1}^{n}(reference_i-target_i)^2

    Parameters
    ----------
    reference : xarray.DataArray
    target : xarray.DataArray

    Returns
    -------
    mean_square_error: xarray.DataArray
        A data array with the time-series of the mean square error

    """

    non_temporal_dimensions = [d for d in reference.dims if d != "time"]
    return ((target - reference) ** 2).mean(dim=non_temporal_dimensions)


def root_mean_square_error(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:
    r"""
    Compute the root mean square error:

    .. math::
        \sqrt{\frac{1}{n}\sum_{i=1}^{n}(reference_i-target_i)^2}

    Parameters
    ----------
    reference : xarray.DataArray
    target : xarray.DataArray

    Returns
    -------
    root_mean_square_error: xarray.DataArray
        A data array with the time-series of the root mean square error

    """
    return mean_square_error(reference, target) ** .5


# First we start defining few functions used in this file
def inter_quartile_range(array: xarray.DataArray) -> xarray.DataArray:
    """
    Returns the inter quartile range of a data array
    :param array:
    :return:
    """
    return array.quantile(0.75) - array.quantile(0.25)


def value_range(array: xarray.DataArray) -> xarray.DataArray:
    """
    Returns the value range of a data array
    :param array:
    :return:
    """
    return array.max() - array.min()


def normalized_root_mean_square_error(reference: xarray.DataArray, target: xarray.DataArray,
                                      method: str = "iqr") -> xarray.DataArray:
    r"""
    Normalized RMSE with two normalization methods, absolute range and inter quartile range
    (less sensitive to outliers)

    .. math::
        \sqrt{\frac{1}{n}\sum_{i=1}^{n}(reference_i-target_i)^2}/range

    Parameters
    ----------
    reference : xarray.DataArray
    target : xarray.DataArray
    method: str
        select between the iqr (inter quartile range) or the range (absolute range). Default is "iqr"

    Returns
    -------
    root_mean_square_error: xarray.DataArray
        A data array with the time-series of the root mean square error

    """
    if method == "iqr":
        normalization_range = inter_quartile_range(reference)
    elif method == "range":
        normalization_range = value_range(reference)
    else:
        raise NotImplementedError

    if normalization_range != 0.:
        return root_mean_square_error(reference, target) / normalization_range
    else:
        return root_mean_square_error(reference, target)


def normalized_root_mean_square_error_index(reference: xarray.DataArray, target: xarray.DataArray,
                                            method="iqr") -> xarray.DataArray:
    """
    Description: Normalized Root Mean Square Error as index.
                For values that are really close to 0, we return the -log10 of the value, to have a better feeling on
                how close to the original value it is.

    Functions
    :param reference:
    :param target:
    :param method: normalization method used. Options are 'iqr' or 'range'
    :return:
    """
    nrmse = normalized_root_mean_square_error(reference=reference, target=target, method=method)
    return xarray.where(nrmse > 0, - np.log10(nrmse), np.inf)
