import numpy as np
import xarray
import dask.array as da


def __normalize(variable):
    """
    Scale a variable to mean zero and standard deviation 1

    Parameters
    ----------
    variable : xarray.DataArray or np.ndarray

    Returns
    -------
    xarray.DataArray or np.ndarray
    """
    mean = variable.mean()
    std = variable.std()
    result = (variable - mean) / std
    return result


def prepare(*variables, **kwargs):
    """

    Parameters
    ----------
    variables : xarray.DataArray or np.ndarray
            one or more variables which should be used for the clustering.

    **kwargs
            *ens_dim* : int or string
                index or name of the dimension along which the clustering should be performed

    Returns
    -------
    np.ndarray
            2d-array with dimensions (ensemble, feature) where ensemble is the dimension along
            which the clustering should be performed and feature is the product of grid points and number of
            variables.
    """

    # 1. step: check the shape of all input variables
    shape = variables[0].shape
    for ivar, one_var in enumerate(variables):
        if one_var.shape != shape:
            raise ValueError("all input variables have to have the same shape! First variable: %s, variable %d: %s"
                             % (shape, ivar+1, one_var.shape))

    # 2. calculate the shape of the result
    ens_dim = kwargs.get("ens_dim", 0)
    feature_shape = list(shape)
    del feature_shape[ens_dim]
    feature_shape = (np.product(feature_shape),)

    # 3. flatten all arrays
    var_list = []
    for one_var in variables:
        # is it an xarray? use directly the data array
        if isinstance(one_var, xarray.DataArray):
            one_var = one_var.data
        # reorder to bring the ensemble dimension to the front
        if ens_dim != 0:
            one_var = np.moveaxis(one_var, ens_dim, 0)
        # flatten all other dimensions
        if one_var.ndim > 2:
            one_var = np.reshape(one_var, (one_var.shape[0],) + feature_shape)
        var_list.append(one_var)

    # 4. normalize all arrays
    var_list = list(map(lambda x: __normalize(x), var_list))

    # Finally concatenate all variables
    return da.concatenate(var_list, axis=1)
