"""
Computation of basic ensemble statistics
"""
import xarray
from enstools.misc import get_ensemble_dim
from enstools.core.parallelisation import rechunk_arguments


@rechunk_arguments({"time": 1})
def ensemble_stat(dataset, stat=["mean", "min", "max", "std"], dim=None):
    """
    Compute ensemble mean and standard deviation from an input array or dataset.

    Parameters
    ----------
    dataset : xarray.DataArray or xarray.Dataset
            input data with ensemble dimension

    stat : list
            list auf standard statistics to compute. Implemented are "mean", "min", "max", "std".

    dim : str
            name of the ensemble dimension. If None, the name is determined automatically.

    Returns
    -------
    tuple :
            (mean, min_, max_, std). Tuple with one item per item in argument stat.
    """
    # check the input data
    if not isinstance(dataset, xarray.core.dataarray.DataArray) and not isinstance(dataset, xarray.core.dataset.Dataset):
        raise ValueError("the input to ensemble_stat has to be an xarray array or dataset!")

    # try to find the ensemble dimension
    if dim is None:
        dim = get_ensemble_dim(dataset)
        if dim is None:
            raise ValueError("ensemble_stat: unable to auto-detect the name of the ensemble dimension")

    # perform the calculation
    results = []
    for one_stat in stat:
        if one_stat == "mean":
            results.append(dataset.mean(dim=dim))
        if one_stat == "min":
            results.append(dataset.min(dim=dim))
        if one_stat == "max":
            results.append(dataset.max(dim=dim))
        if one_stat == "std":
            results.append(dataset.std(dim=dim))

    # return the result
    return tuple(results)
