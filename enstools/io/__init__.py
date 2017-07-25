"""
Reading and Writing of meteorological data
"""
import xarray


def read(files, **kwargs):
    """
    Read one or more input files

    Parameters
    ----------
    files : str or list of str
            One or more file names to read

    **kwargs
            all arguments accepted by xarray.open_dataset() of xarray.open_mfdataset()

    Returns
    -------
    xarray.Dataset
            in-memory representation of the content of the input files
    """

    # only one filename if given, open it directly
    if isinstance(files, str):
        ds = xarray.open_dataset(files, **kwargs)
        return ds

    # read an array of files
    if isinstance(files, (list, tuple)):
        ds = xarray.open_mfdataset(files, **kwargs)
        return ds

    # unexpected input
    raise TypeError("unexpected type of argument 'files': %s" % type(files))
