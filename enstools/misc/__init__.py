import os
import logging
import bz2
import six
import xarray
import numpy as np
from numba import jit
import dask
from shutil import copyfileobj


if six.PY3:
    from urllib.request import urlretrieve, urlopen
else:
    from urllib import urlretrieve


def download(url, destination, uncompress=True):
    """
    Download a file from the web

    Parameters
    ----------
    url : string
            the web address

    destination : string
            path to store the file. The file will only be downloaded once.

    uncompress : bool
            if True and if the name ends with '.bz2', it will
            automatically be uncompressed.
    """
    if destination.endswith(".bz2") and uncompress:
        destination_intern = destination[:-4]
    else:
        destination_intern = destination

    if os.path.exists(destination_intern):
        logging.warning("file not downloaded because it is already present: %s" % url)
        return destination_intern

    # download
    logging.info("downloading %s ..." % os.path.basename(destination))
    logging.debug("from: %s" % url)
    fn, hd = urlretrieve(url, destination)

    # uncompress
    if destination.endswith(".bz2") and uncompress:
        # uncompress
        bfile = bz2.BZ2File(destination)
        dfile = open(destination_intern, "wb")
        dfile.write(bfile.read())
        dfile.close()
        bfile.close()
        # delete compressed
        os.remove(destination)

    # return the name if the downloaded file
    return destination_intern


@jit(["b1(f4[:],f4[:],f4,f4)", "b1(f8[:],f8[:],f8,f8)"], nopython=True)
def point_in_polygon(polyx, polyy, testx, testy):
    """
    check whether or not a given coordinate is inside or outside of a polygon

    Parameters
    ----------
    polyx : np.ndarray
            x-coordinates of the polygon

    polyy : np.ndarray
            y-coordinates of the polygon

    testx : float
            x-coordinate of the point

    testy : float
            y-coordinate of the point

    Returns
    -------
    bool
            True, if the point is inside of the polygon
    """
    res = False
    j = polyx.shape[0] - 1
    for i in range(polyx.shape[0]):
        if ((polyy[i] > testy) != (polyy[j] > testy)) \
                and (testx < (polyx[j] - polyx[i]) * (testy - polyy[i]) / (polyy[j] - polyy[i]) + polyx[i]):
            res = not res
        j = i
    return res


def spherical2cartesian(lon, lat, radius=6371229.0):
    """
    calculate cartesian 3d coordinates from spherical coordinates

    Parameters
    ----------
    lon: np.array
            longitude in radian

    lat: np.array
            latitude in radian

    radius: float
            radius of the earth.

    Returns
    -------
    np.array (n, 3)
            cartesian coordinates (x, y, z)
    """
    # create result array
    result = np.empty((lon.shape[0], 3))
    _lat = lat - np.pi / 2.0
    _lon = lon + np.pi
    result[:, 0] = radius * np.sin(_lat) * np.cos(_lon)
    result[:, 1] = radius * np.sin(_lat) * np.sin(_lon)
    result[:, 2] = radius * np.cos(_lat)
    return result


def generate_coordinates(res, grid="regular", lon_range=[-180, 180], lat_range=[-90, 90], unit="degrees"):
    """
    Generate grid coordinates for different types of grids. Currently only regular grids are implemented.

    Parameters
    ----------
    res : float
            resolution of the grid to generate

    grid : {'regular'}
            type of grid to generate. Currently only regular grids are supported
    
    lon_range : list or tuple
            range of longitudes the new grid should cover. Default: -180 to 180

    lat_range : list or tuple
            range of latitudes the new grid should cover. Default: -90 to 90

    unit: {'degrees', 'radians'}

    Returns
    -------
    lon, lat : xarray.DataArray
            tuple of coordinate arrays

    Examples
    --------
    >>> lon, lat = generate_coordinates(20.0, "regular")
    >>> lon
    <xarray.DataArray 'lon' (lon: 18)>
    array([-180., -160., -140., -120., -100.,  -80.,  -60.,  -40.,  -20.,    0.,
             20.,   40.,   60.,   80.,  100.,  120.,  140.,  160.])
    Dimensions without coordinates: lon
    Attributes:
        units:    degrees_east
    >>> lat
    <xarray.DataArray 'lat' (lat: 9)>
    array([-80., -60., -40., -20.,   0.,  20.,  40.,  60.,  80.])
    Dimensions without coordinates: lat
    Attributes:
        units:    degrees_north
    """
    if grid == "regular":
        lon = xarray.DataArray(np.arange(lon_range[0], lon_range[1], res), dims=("lon",), name="lon", attrs={"units": "degrees_east"})
        lat = xarray.DataArray(np.arange(lat_range[0]+res/2.0, lat_range[1], res), dims=("lat",), name="lat", attrs={"units": "degrees_north"})
        if unit == "degrees":
            lon.attrs["units"] = "degrees_east"
            lat.attrs["units"] = "degrees_north"
        elif unit == "radians":
            lon.attrs["units"] = "radians"
            lat.attrs["units"] = "radians"
            lon *= np.pi / 180.0
            lat *= np.pi / 180.0
        else:
            raise ValueError(f"unsupported unit: {unit}")
    else:
        raise ValueError("unsupported grid type: '%s'" % grid)

    return lon, lat


def swapaxis(array, a1, a2):
    """
    move an axis of an array from one position to another. This function belongs to the numpy library but is not
    implemented for xarray.DataArray

    Parameters
    ----------
    array : xarray.DataArray or np.ndarray
            array to manipulate

    a1 : int
            first axis

    a2 : int
            second axis

    Returns
    -------
    xarray.DataArray or np.ndarray
            array with swap axis
    """
    dims = np.arange(array.ndim)
    dims[a1] = a2
    dims[a2] = a1
    dims = tuple(dims)
    if isinstance(array, xarray.DataArray):
        dims = tuple(map(lambda x:array.dims[x], dims))
    return array.transpose(*dims)


def has_ensemble_dim(ds):
    """
    check whether or not a dataset or xarray has already an ensemble dimension

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    bool
    """
    return get_ensemble_dim(ds) is not None


def get_ensemble_dim(ds):
    """
    get the name of the ensemble dimension from a dataset or array

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str or None :
            if no ensemble dimension was found, None is returned.
    """
    ens_names = ["ens", "ensemble", "member", "members"]
    for ens_name in ens_names:
        if ens_name in ds.dims:
            logging.debug("get_ensemble_dim: found name '%s'" % ens_name)
            return ens_name
    return None


def set_ensemble_member(ds, member):
    """
    set the number of the ensemble member. The Dataset is modified inplace.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
            Array or Dataset with ensemble dimension and only one member
    """
    # get the name of the ensemble dimension
    ens_dim = get_ensemble_dim(ds)

    # if we dont have one, add one
    if ens_dim is None:
        add_ensemble_dim(ds, member)
    else:
        if ds.coords[ens_dim].size == 1:
            ds.coords[ens_dim] = [member]
        else:
            logging.debug("set_ensemble_member: ds has more than one member. Doing nothing!")


def get_time_dim(ds):
    """
    get the name of the time dimension from a dataset or array

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str or None :
            if no ensemble dimension was found, None is returned.
    """
    time_names = ["time", "Time", "times", "Times"]
    for time_name in time_names:
        if time_name in ds.dims:
            logging.debug("get_time_dim: found name '%s'" % time_name)
            return time_name
    return None


def has_dask_arrays(dataset):
    """
    check whether or not a dataset contains dask arrays

    Parameters
    ----------
    dataset : xarray.Dataset

    Returns
    -------
    bool :
            True if dask arrays are found
    """
    has_dask = False
    for varname, var in six.iteritems(dataset.variables):
        if isinstance(var.data, dask.array.core.Array):
            has_dask = True
            break
    return has_dask


def add_ensemble_dim(ds, member, inplace=True):
    """
    create an ensemble dimension with in the dataset

    Parameters
    ----------
    ds : xarray.Dataset

    member : int
            number of the ensemble member

    inplace : bool
            modify the dataset directly?

    Returns
    -------
    xarray.Dataset:
            A copy of the dataset expanded by the ensemble dimension or the expanded dataset
            if the expansion was done inplace.
    """
    # create a lazy copy of the dataset
    if inplace:
        new_ds = ds
    else:
        new_ds = ds.copy()

    # loop over all data variables.
    # those with time dimension are extended behind the time dimension, others at the front
    for one_name, one_var in six.iteritems(ds.data_vars):
        if is_additional_coordinate_variable(one_var):
            continue
        if len(one_var.dims) > 0 and one_var.dims[0] == "time":
            new_ds[one_name] = one_var.expand_dims("ens", 1)
        else:
            new_ds[one_name] = one_var.expand_dims("ens")
    new_ds.coords["ens"] = [member]

    # remove the ensemble_member attribute if present. it if only intented for datasets without ensemble dimension
    if "ensemble_member" in new_ds.attrs:
        del new_ds.attrs["ensemble_member"]
    return new_ds


def is_additional_coordinate_variable(var):
    """
    check if a variable belongs to a list of known constant coordinate variables

    Parameters
    ----------
    var : xarray.DataArray

    Returns
    -------
    bool :
           True = variable is not different for ensemble members
    """

    # list of excluded variables from different models
    excluded = {
        # COSMO variables
        "time_bnds": ("time", "bnds"),
        "slonu": ("rlat", "srlon"),
        "slatu": ("rlat", "srlon"),
        "slonv": ("srlat", "rlon"),
        "slatv": ("srlat", "rlon"),
        "vcoord": ("level1",),
        "soil1_bnds": ("soil1", "bnds"),
        "rotated_pole": (),
        "height_2m": (),
        "height_10m": (),
        "height_toa": (),
        "wbt_13c": (),
    }

    # variable is in excluded list?
    if var.name in excluded and excluded[var.name] == var.dims:
        return True
    else:
        return False


def first_element(array):
    """
    returns the first element of an array or the value of an scalar.

    Parameters
    ----------
    array : xarray.DataArray
            array with 0 or more dimensions.

    Returns
    -------
    float :
            first value.
    """
    if array.size > 1:
        return float(array[0])
    else:
        return float(array)


def count_ge(array, th=0):
    """
    count the number of values above a given threshold (>=)
    Parameters
    ----------
    array : xarray.DataArray and numpy.ndarray
            an array with arbitrary number of dimensions

    th : float
            threshold to test the array for.

    Returns
    -------
    int :
            number of values greater then or equal the threshold.
    """
    if type(array) == xarray.DataArray:
        return __count_ge(array.data, th)
    else:
        return __count_ge(array, th)


@jit(nopython=True)
def __count_ge(array, th):
    """
    numba implementation of count_ge
    """
    result = 0
    for i in range(array.size):
        if array.flat[i] >= th:
            result += 1
    return result


def concat(files, out_filename):
    """
    Concatenates multiple files to one.
    Parameters
    ----------
    files: list
        The list of the files to concat
    out_filename: str
        The name (with destination) of the merged file

    Returns
    -------

    """
    with open(out_filename, "wb") as out:
        for filename in files:
            with open(filename, "rb") as file:
                copyfileobj(file, out)


def bytes2human(n, format='%(value).1f %(symbol)s', symbols='iec'):
    """
    Convert n bytes into a human readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs
    """
    SYMBOLS = {
        'customary': ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'),
        'customary_ext': ('byte', 'kilo', 'mega', 'giga', 'tera', 'peta', 'exa', 'zetta', 'iotta'),
        'customary_with_B': ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'),
        'iec': ('Bi', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'),
        'iec_ext': ('byte', 'kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi', 'zebi', 'yobi'),
        'iec_with_B': ('Bi', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'),
    }

    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    if "iec" in symbols:
        base = 1024
    else:
        base = 1000
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = base ** i
    for symbol in reversed(symbols):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)



