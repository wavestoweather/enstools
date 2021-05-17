import xarray
from xarray.backends.netCDF4_ import NetCDF4DataStore

from enstools.misc import has_dask_arrays
from .file_type import get_file_type
from .encoding import set_encoding, encoding_description
import dask.array
import six
from distutils.version import LooseVersion


def write(ds, filename, file_format=None, compression="lossless", compute=True):
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
            
    compression : string
            Used to specify the compression mode and optionally additional arguments.
            
            To apply lossless compression we can just use:
                "lossless"
            Or we can select the backend and the compression level using the following syntax:
                "lossless:backend:compression_level"
            The backend can be one of:
                    'blosclz' 
                    'lz4' (default)
                    'lz4hc'
                    'snappy'
                    'zlib'
                    'zstd'
            and the compression level must be an integer from 1 to 9 (default is 9).
            Few examples:
                "lossless:zstd:4"
                "lossless:lz4:9"
                "lossless:snappy:1"
            Using "lossless" without additional arguments would be equivalent to "lossless:lz4:9"
            
            For lossy compression, we might be able to pass more arguments:
                "lossy"
            The lossy compressors available will be:
                'zfp'
                'sz'
            For ZFP we have different compression methods available:
                'rate'
                'accuracy'
                'precision'
            Each one of this methods require an additional parameter: the rate, the precision or the accuracy.
            The examples would look like:
                'lossy:zfp:accuracy:0.2'
                'lossy:zfp:rate:4'
            Another option would be to pass a configuration file as argument. (json?)
    compute : bool 
            Dask delayed feature. Set to true to delay the file writting.
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
    encoding = set_encoding(ds, compression)
    # In case of using an encoding, we'll add an attribute to the file indicating that the file has been compressed.
    if encoding is not None:
        descriptions = encoding_description(encoding)
        for variable, description in descriptions.items():
            ds[variable].attrs["compression"] = description

    # write a netcdf file
    if selected_format == "NC":
        #"""
        print(encoding)
        task = ds.to_netcdf(filename, engine="h5netcdf", encoding=encoding, compute=compute)
        # ds.to_netcdf(filename)
        return task
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

