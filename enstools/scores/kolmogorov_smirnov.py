from typing import Union

import numpy
import xarray
from scipy.stats import ks_2samp
from .fix_attributes import fix_attributes
from enstools.core.errors import EnstoolsError


def kolmogorov_smirnov(reference: xarray.DataArray, target: xarray.DataArray, to_return="pvalue")\
        -> Union[xarray.DataArray, xarray.Dataset]:
    """
    Description: Compute the KS statistic of the full domain and return the pvalue .

    Functions
    :param reference:
    :param target:
    :param to_return: select between returning the 'pvalue', the 'statistic' as a data Arrays or 'both' as a Dataset
    :return:
    """

    ks_statistic = reference.copy(deep=True)

    non_temporal_dimensions = [d for d in ks_statistic.dims if d != "time"]
    selector = {dim: reference[dim][0] for dim in non_temporal_dimensions}
    ks_statistic = reference.sel(selector)

    ks_pvalue = ks_statistic.copy(deep=True)

    if "time" in reference.dims:
        times = reference["time"]
        ks_results = [
            ks_2samp(target.sel(time=t).values.ravel(), reference.sel(time=t).values.ravel(), alternative='two-sided',
                     mode='auto') for t in times]
        statistic = [ks.statistic for ks in ks_results]
        p_value = [ks.pvalue for ks in ks_results]

    else:
        statistic, p_value = ks_2samp(target.ravel(), reference.ravel(), alternative='two-sided', mode='auto')

    # Put the results of the KS
    ks_pvalue.values = p_value
    ks_statistic.values = statistic

    # Fix attributes
    ks_statistic = fix_attributes(ks_statistic, "ks_statistic")
    ks_pvalue = fix_attributes(ks_pvalue, "ks_pvalue")

    # Combine data arrays in dataset
    result = xarray.Dataset(dict(statistic=ks_statistic, pvalue=ks_pvalue))
    if to_return in ["pvalue", "statistic"]:
        return result[to_return]
    elif to_return == "both":
        return result
    else:
        raise EnstoolsError(
            f"the parameter 'to_return' for kolmogorov_smirnov accepts the options 'pvalue', 'statistic' or 'both'. "
            f"{to_return!r} was provided.")


def kolmogorov_smirnov_index(reference: xarray.DataArray, target: xarray.DataArray, to_return="pvalue")\
        -> Union[xarray.DataArray, xarray.Dataset]:
    """
    Description: Compute the KS statistic of the full domain and return the -log10(1 - pvalue) .

    Functions
    :param reference:
    :param target:
    :param to_return: select between returning the 'pvalue', the 'statistic' as a data Arrays or 'both' as a Dataset
    :return:
    """
    ks = kolmogorov_smirnov(reference, target)
    return xarray.where(ks < 1.0, - numpy.log10(1 - ks), numpy.inf)
