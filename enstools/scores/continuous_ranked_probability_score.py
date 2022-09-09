import numpy as np
import xarray
from .fix_attributes import fix_attributes
from enstools.core.errors import EnstoolsError


def continuous_ranked_probability_score(reference: xarray.DataArray, target: xarray.DataArray,
                                        ensemble_dimension: str = "ens") -> xarray.DataArray:
    """
    Description: My implementation of the Continuous Ranked Probability Score.

    Functions
    :param reference:
    :param target:
    :param ensemble_dimension: name of the ensemble dimension, usually is 'ens' or 'member'
    :return:
    """
    if ensemble_dimension not in reference.dims:
        raise EnstoolsError(f"Trying to compute CRPS on a dataset that doesn't contain ensemble_dimension:{ensemble_dimension}")
    # Make a copy of the reference data array to store the results in it.
    result = reference.copy(deep=True)
    result = result.sel(ens=1)

    # Reorder coordinates to put the ensemble dimension in first place
    reference = reference.transpose(ensemble_dimension, ...)
    target = target.transpose(ensemble_dimension, ...)

    # Get ensemble sizes
    reference_ensemble_size = reference[ensemble_dimension].size
    target_ensemble_size = target[ensemble_dimension].size

    ensemble_index = reference.dims.index(ensemble_dimension)

    # We need to find the minimum and maximum values for each grid point for both datasets
    ref_min, ref_max = reference.min(dim=ensemble_dimension), reference.max(dim=ensemble_dimension)
    trg_min, trg_max = target.min(dim=ensemble_dimension), target.max(dim=ensemble_dimension)

    # Get minimums and maximums for each grid point.
    _min = np.stack((ref_min, trg_min)).min(axis=0)
    _max = np.stack((ref_max, trg_max)).max(axis=0)

    # For each grid point, create a linear space that will cover the full range of values of the two arrays
    x = np.linspace(_min, _max, reference_ensemble_size)

    # Two empty arrays are created, will be used to count the number of data values that are below certain threshold
    # to estimate the ECDF in both arrays using the same reference
    value_occurrence_ref = np.zeros(reference.shape)
    value_occurrence_trg = np.zeros(reference.shape)

    # Load the data to avoid problems with dask
    reference.load()
    target.load()


    for member in range(reference["ens"].size):
        selection = tuple([member if dim == ensemble_dimension else slice(None) for dim in reference.dims])
        value_occurrence_ref[selection] = (reference < x[member]).sum(dim=ensemble_dimension)
        value_occurrence_trg[selection] = (target < x[member]).sum(dim=ensemble_dimension)

    # We normalize by the size of each ensemble
    value_occurrence_ref = value_occurrence_ref / reference_ensemble_size
    value_occurrence_trg = value_occurrence_trg / target_ensemble_size

    # Compute error as the area between the ECDF
    area = np.trapz(np.abs(value_occurrence_ref - value_occurrence_trg), x, axis=ensemble_index)

    result.values = area

    # Add crps to attributes and delete obsolete ones:
    result = fix_attributes(result, suffix="crps")

    # Return error average
    return result
