import xarray


def positivity(reference: xarray.DataArray, target: xarray.DataArray) -> xarray.DataArray:
    """
    Description:
    This metric evaluates if reference values are strictly positive and checks whether this is
    maintained in the target dataset.
    It returns 1.0 if the reference and the target have the same positivity and -1.0 otherwise.
    These values are arbitrary.


    Functions
    :param reference:
    :param target:
    :return:
    """
    if (reference >= 0.0).all() == (target >= 0.0).all():
        return xarray.DataArray(1.0)
    else:
        return xarray.DataArray(-1.0)

