import xarray
import numpy
from scipy.stats import pearsonr


def pearsonr_wrapper(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """
    Small wrapper for the pearsonr function, converting the input arrays to 1D arrays
    """
    corr, _ = pearsonr(a.ravel(), b.ravel())
    return corr


def pearson_correlation(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:

    r"""
    It returns the Pearson's correlation.
    Uses `implementation from scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html>`_.

    Parameters
    ----------
    reference : xarray.DataArray
    target : xarray.DataArray

    Returns
    -------
    pearson_correlation: xarray.DataArray
        A data array with the time-series of the pearson correlation

    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    core_dims = [d for d in reference.dims if d != "time"]

    # Load data
    reference.load()
    target.load()

    return xarray.apply_ufunc(pearsonr_wrapper, reference, target,
                              #
                              vectorize=True,
                              input_core_dims=[core_dims, core_dims],
                              output_core_dims=[[]],
                              )


def pearson_correlation_index(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:
    """
    Description:
        It returns the Pearson's correlation index. Uses implementation from scipy.
        This metrics is expected to be useful when te values are close to 1.

    Functions
    :param reference:
    :param target:
    :return:
    """
    correlation = pearson_correlation(reference, target)

    # The function pearson_correlation from scipy returns a NaN if one of the input arrays is constant
    # which can happen quite often if the compression is extreme.
    # We'll instead return a correlation of 0.
    correlation = correlation.fillna(0)

    return xarray.where(correlation == 1.0, numpy.inf, - numpy.log10(1-correlation))

