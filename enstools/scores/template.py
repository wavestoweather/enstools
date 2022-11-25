import xarray


def metric(reference: xarray.DataArray, target: xarray.DataArray,
           extra_argument: str = "Ideally extra arguments must have a default value") -> xarray.DataArray:
       """
       Description: A meaningful description of the metric with references if necessary would be nice.

       Parameters
       ----------
       reference: xarray.DataArray

       target: xarray.DataArray

       Returns
       -------

       metric: xarray.DataArray
       """
       print(extra_argument)
       return target - reference
