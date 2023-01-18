import logging
from distutils.version import LooseVersion
from os import rename
from typing import Union

import dask.array
import six
import xarray
from xarray.backends.netCDF4_ import NetCDF4DataStore
from pathlib import Path

try:
    from enstools.encoding.api import DatasetEncoding

    encoding_available = True
except ModuleNotFoundError:
    encoding_available = False

try:
    import enstools.compression

    compression_available = True
except ModuleNotFoundError:
    compression_available = False

from .file_type import get_file_type


def write(ds: Union[xarray.Dataset, xarray.DataArray],
          file_path: Union[str, Path],
          file_format: Union[str, None] = None,
          compression: Union[str, dict, None] = None,
          compute: bool = True,
          engine: str = "h5netcdf",
          format: str = "NETCDF4",
          ):
    """
    write a xarray dataset to a file

    Parameters
    ----------
    ds : xarray.Dataset
            the dataset to store

    file_path : string or Path
            the file to create

    file_format : {'NC'}
            string indicating the format to use. if not specified, the file extension if used.
            
    compression : string
            Used to specify the compression mode and optionally additional arguments.
            The parameter follows the rules defined in enstools-encoding.
            (https://gitlab.physik.uni-muenchen.de/w2w/enstools-encoding.git)
            
            To apply lossless compression we can just use:
                "lossless"
            Or we can select the backend and the compression level using the following syntax:
                "lossless,backend,compression_level"
            The backend can be one of:
                    'blosclz' 
                    'lz4' (default)
                    'lz4hc'
                    'snappy'
                    'zlib'
                    'zstd'
            and the compression level must be an integer from 1 to 9 (default is 9).
            Few examples:
                "lossless,zstd,4"
                "lossless,lz4,9"
                "lossless,snappy,1"
            Using "lossless" without additional arguments would be equivalent to "lossless,lz4,9"
            
            For lossy compression, we might be able to pass more arguments:
                "lossy"
            The lossy compressors available will be:
                'zfp'
                'sz'
            For ZFP we have different compression methods available:
                'rate'
                'accuracy'
                'precision'
            For SZ we have also different compression methods available:
                'abs'
                'rel'
                'pw_rel'
            Each one of this methods require an additional parameter: the rate, the precision or the accuracy.
            The examples would look like:
                'lossy,zfp,accuracy,0.2'
                'lossy,zfp,rate,4'
            There are also few features that target datasets with multiple variables.
            One can write a different specification for different variables by using a
            list of space separated specifications:

            'var1:lossy,zfp,rate,4.0 var2:lossy,sz,abs,0.1'

            For more details see the corresponding documentation.

            Another option would be to pass the path to a YAML configuration file as argument.
    compute : bool 
            Dask delayed feature. Set to true to delay the file writing.
    """

    file_path = Path(file_path).resolve()

    # if ds is a DataVariable instead of a Dataset, then convert it
    if isinstance(ds, xarray.DataArray):
        ds = ds.to_dataset()

    # select the type of file to create
    valid_formats = ["NC"]
    if file_format is not None:
        selected_format = file_format
    else:
        selected_format = get_file_type(file_path.name, only_extension=True)
    if selected_format not in valid_formats:
        raise ValueError("the format '%s' is not (yet) supported!" % selected_format)

    if compression is not None:
        help_message = "To use the compression argument please install enstools-encoding:\n" \
                       "pip install enstools-encoding"
        assert compression_available, ModuleNotFoundError(help_message)
        # If a compression variable has been provided, define the proper encoding for HDF5 filters:
        dataset_encoding = DatasetEncoding(dataset=ds, compression=compression)
        dataset_encoding.add_metadata()
    else:
        dataset_encoding = None

    if format == "NETCDF4_CLASSIC" and engine == "h5netcdf":
        raise AssertionError(
            f"enstools.io.write:: Format {format} and engine {engine} are not compatible."
            f"If format {format} is needed, the suggested engine is NETCDF4")

    if engine == "netcdf4":
        if compression not in [None, "default"]:
            logging.warning("Using netcdf4 engine. Setting encoding to None")
            dataset_encoding = None

    # write a netcdf file
    if selected_format == "NC":
        # We can do the trick of changing the name to filename.tmp and changing it back after the process is completed
        # but only if we execute the task here ( i.e. compute==True).
        if compute:
            final_filename = file_path
            file_path = f"{file_path}.tmp"
        task = ds.to_netcdf(file_path, engine=engine, encoding=dataset_encoding, compute=compute, format=format)
        if compute:
            rename(file_path, final_filename)
        return task

        """
        #Old workaround (need to be sure that its not needed anymore)
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
