import xarray
from xarray.backends.netCDF4_ import NetCDF4DataStore

from enstools.misc import has_dask_arrays
from .file_type import get_file_type
import dask.array
import six
from distutils.version import LooseVersion


def write(ds, filename, file_format=None):
    """
    write a xarray dataset to a file

    Parameters
    ----------
    ds : xarray.Dataset
            the dataset to store

    filename : string
            the file to create

    file_format : {'NC'}
            string indicating the format to use. if not specified, the file extension if used.
    """
    # select the type of file to create
    valid_formats = ["NC"]
    if file_format is not None:
        selected_format = file_format
    else:
        selected_format = get_file_type(filename, only_extension=True)
    if selected_format not in valid_formats:
        raise ValueError("the format '%s' is not (yet) supported!" % selected_format)

    # write a netcdf file
    if selected_format == "NC":
        # are there dask arrays in the dataset? if so, write the file variable by variable
        if not has_dask_arrays(ds):
            ds.to_netcdf(filename)
        else:
            __to_netcdf(ds, filename)


def __to_netcdf(ds, filename):
    """
    workaround for bug if dask arrays

    Parameters
    ----------
    ds : xarray.Dataset
            dataset to store

    filename : string
            name of the new file
    """

    # store the complete file without the dask arrays
    ds_copy = ds.copy(deep=False)
    dask_variables = []
    for varname, var in six.iteritems(ds.data_vars):
        if isinstance(var.data, dask.array.core.Array):
            dask_variables.append(varname)
    for one_var in dask_variables:
        del ds_copy[one_var]
    ds_copy.to_netcdf(filename)

    # open it again and add the dask arrays
    if LooseVersion(xarray.__version__) > LooseVersion('0.9.6'):
        nc = NetCDF4DataStore.open(filename, "a")
    else:
        nc = NetCDF4DataStore(filename, "a")

    # add the variables
    for varname, var in six.iteritems(ds.data_vars):
        if isinstance(var.data, dask.array.core.Array):
            nc.store({varname: var.compute()}, {})

    # close the file
    nc.close()

