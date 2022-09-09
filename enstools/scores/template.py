import xarray


def metric(reference: xarray.DataArray, target: xarray.DataArray,
           extra_argument: str = "Ideally extra arguments must have a default value") -> xarray.DataArray:
    """
    Description: A meaningful description of the metric with references if necessary would be nice.

    Functions
    :param reference:
    :param target:
    :param extra_argument:
    :return:
    """
    print(extra_argument)
    return target - reference
