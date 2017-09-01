from collections import OrderedDict
import xarray
import dask
import six
from multipledispatch import dispatch
import glob
from .file_type import get_file_type
try:
    from .eccodes import read_grib_file
except ImportError:
    pass


@dispatch(six.string_types)
def read(filename, **kwargs):
    """
    Read one or more input files

    Parameters
    ----------
    filename : str
            name of one individual file of unix shell-like file name pattern for multiple files

    **kwargs
            all arguments accepted by xarray.open_dataset() of xarray.open_mfdataset()

    Returns
    -------
    xarray.Dataset
            in-memory representation of the content of the input file(s)
    """
    # is the filename a pattern?
    if "*" in filename or "?" in filename or "[" in filename or "{" in filename:
        files = glob.glob(filename)
        if len(files) > 1:
            return read(files, **kwargs)

    # read the input file
    return __open_dataset(filename)


@dispatch((list, tuple))
def read(filenames, **kwargs):
    """
    Read multiple input files

    Parameters
    ----------
    filename : list of str or tuple of str
            names of individual files or filename pattern

    **kwargs
            all arguments accepted by xarray.open_dataset() of xarray.open_mfdataset()

    Returns
    -------
    xarray.Dataset
            in-memory representation of the content of the input file(s)
    """
    # create at first a list of all input files
    datasets = []
    # loop over all input files and create delayed read objects
    for filename in filenames:
        datasets.append(dask.delayed(read)(filename))
    # read all the files in parallel
    datasets = dask.compute(*datasets, traverse=False, get=dask.multiprocessing.get)

    # find common coordinates and sort variables according to those coordinates
    # get all coordinates
    all_coords = set()
    for ds in datasets:
        for coord in ds.coords.keys():
            if coord not in all_coords:
                all_coords.add(coord)
    # filter for coordinates present in all files
    coords_in_all_files = OrderedDict()
    for coord in all_coords:
        in_all = True
        ascending = True
        for ds in datasets:
            if coord in ds.coords:
                # multi-dim coords are not sortable
                if ds.coords[coord].ndim > 1:
                    in_all = False
                    break
                if ds.coords[coord].size > 1:
                    if ds.coords[coord][0] > ds.coords[coord][1]:
                        ascending = False
            else:
                in_all = False
                break
        if in_all:
            coords_in_all_files[coord] = ascending
    # sort the dataset by all shared coordinates
    for coord, ascending in six.iteritems(coords_in_all_files):
        datasets = sorted(datasets, key=lambda x:x.coords[coord][0])

    # try to merge the datasets
    result = xarray.auto_combine(datasets)
    return result


def __open_dataset(filename):
    """
    read one input file. the type is automatically determined.

    Parameters
    ----------
    filename : str
            path of the file

    Returns
    -------
    xarray.Dataset
    """

    # get the type of the file
    file_type = get_file_type(filename)
    if file_type is None:
        raise ValueError("unable to guess the type of the input file '%s'" % filename)

    if file_type == "NC":
        return xarray.open_dataset(filename, chunks={})
    elif file_type == "GRIB":
        return read_grib_file(filename, debug=False)
    else:
        raise ValueError("unknown file type '%s' for file '%s'" % (file_type, filename))
