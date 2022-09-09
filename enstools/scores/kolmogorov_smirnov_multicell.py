import xarray
from scipy.stats import ks_2samp
import numpy as np
from typing import List
from .fix_attributes import fix_attributes


def kolmogorov_smirnov_multicell(reference: xarray.DataArray, target: xarray.DataArray, number_of_cells: int = 100,
                                 ensemble_dimension="ens") \
        -> xarray.DataArray:
    """
    Description:     Compute the Kolmogorov-Smirnov statistic in a random sample of cells.

    Functions
    :param reference:
    :param target:
    :param number_of_cells
    :param ensemble_dimension name of the ensemble dimension, usually 'ens' or 'member'
    :return:
    """
    # Even if we are selecting only few cells, it might be convenient to just load the datasets to speedup things:
    reference.load()
    target.load()
    # Create a data array from the reference.
    result = reference.copy(deep=True)
    non_temporal_dimensions = [d for d in result.dims if d != "time"]
    selector = {dim: reference[dim][0] for dim in non_temporal_dimensions}
    result = result.sel(selector)

    # Randomly select a certain number of cells
    random_cells = random_select_cells(reference, ensemble_dimension=ensemble_dimension, size=number_of_cells)
    if "time" in reference.dims:
        times = reference["time"]
        statistics_time_series = []
        for t in times:
            statistics = [
                ks_2samp(target.sel(time=t).sel(sl).values.ravel(), reference.sel(time=t).sel(sl).values.ravel(),
                         alternative='two-sided', mode='auto').statistic for sl in random_cells]
            statistics_time_series.append(np.mean(statistics))
        result.values = statistics_time_series
    else:
        statistics = [ks_2samp(target.sel(sl).values.ravel(), reference.sel(sl).values.ravel(),
                               alternative='two-sided', mode='auto').statistic for sl in random_cells]
        result.values = np.mean(statistics)

    result = fix_attributes(result, "ks_multicell")
    return result


def random_select_cell(data_array: xarray.DataArray, ensemble_dimension: str = "ens") -> dict:
    """
    Uses numpy to randomly select a cell.
    Since xarray.DataArray are used, we identify each cell by a dictionary containing a value for each dimension.

    #TODO: Must check if this selection mode is actually slower than using indices.

    :param data_array:
    :param ensemble_dimension:
    :return:
    """
    dimensions_to_ignore = ["time", ensemble_dimension]

    selection = {dim: np.random.choice(data_array[dim]) for dim in data_array.dims if dim not in dimensions_to_ignore}

    return selection


def random_select_cells(data_array: xarray.DataArray, ensemble_dimension: str = "ens", size: int = 1) -> List[dict]:
    """
    Randomly select a certain number of cells
    :param data_array:
    :param ensemble_dimension:
    :param size:
    :return:
    """
    return [random_select_cell(data_array, ensemble_dimension=ensemble_dimension) for _ in range(size)]
