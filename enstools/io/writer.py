import xarray
from xarray.backends.netCDF4_ import NetCDF4DataStore

from enstools.misc import has_dask_arrays
from .file_type import get_file_type
from .encoding import set_encoding
import dask.array
import six
from distutils.version import LooseVersion



def write(ds, filename, file_format=None, compression_mode="lossless"):
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
            
    compressor : string
            one of the available backends for the BLOSC filter:
                    'blosclz' 
                    'lz4' (default)
                    'lz4hc'
                    'snappy'
                    'zlib'
                    'zstd'

    clevel : int 
            integer that indicates the compression level (from 1 to 9). default = 9
    """
    # if ds is a DataVariable instead of a Dataset, then convert it
    if isinstance(ds, xarray.DataArray):
        ds = ds.to_dataset()
    # select the type of file to create
    valid_formats = ["NC"]
    if file_format is not None:
        selected_format = file_format
    else:
        selected_format = get_file_type(filename, only_extension=True)
    if selected_format not in valid_formats:
        raise ValueError("the format '%s' is not (yet) supported!" % selected_format)

        
     

    # Encoding
    encoding = set_encoding(ds, compression_mode)
    # write a netcdf file
    if selected_format == "NC":
        #"""
        try:
            ds.to_netcdf(filename, engine="h5netcdf", encoding=encoding)
            #ds.to_netcdf(filename)
        
        except ValueError as err:
            # Can rise the error or just remove the enconding.
            print(err)
            print("Compression filter is not available!")
            raise(err)
        #"""
        """
        #Old workaround (neet to be sure that its not needed anymore)
        # are there dask arrays in the dataset? if so, write the file variable by variable
        if not has_dask_arrays(ds):
            ds.to_netcdf(filename)
        else:
            __to_netcdf(ds, filename)
        """


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
    print("Its using __to_netcdf!")
    # store the complete file without the dask arrays
    ds_copy = ds.copy(deep=False)
    dask_variables = []
    for varname, var in six.iteritems(ds.variables):
        if isinstance(var.data, dask.array.core.Array):
            dask_variables.append(varname)
    for one_var in dask_variables:
        del ds_copy[one_var]
    ds_copy.to_netcdf(filename, format="NETCDF4_CLASSIC")

    # open it again and add the dask arrays
    if LooseVersion(xarray.__version__) > LooseVersion('0.9.6'):
        nc = NetCDF4DataStore.open(filename, "a")
    else:
        nc = NetCDF4DataStore(filename, "a")

    # add the variables
    for varname, var in six.iteritems(ds.variables):
        if isinstance(var.data, dask.array.core.Array):
            nc.store({varname: var.compute()}, {})

    # close the file
    nc.close()

