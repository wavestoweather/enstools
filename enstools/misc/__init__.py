import os
import logging
import bz2
import six
import xarray
import numpy as np
from numba import jit

if six.PY3:
    from urllib.request import urlretrieve
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
        return

    # download
    logging.info("downloading %s ..." % os.path.basename(destination))
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
    for i in range(j):
        if ((polyy[i] > testy) != (polyy[j] > testy)) \
                and (testx < (polyx[j] - polyx[i]) * (testy - polyy[i]) / (polyy[j] - polyy[i]) + polyx[i]):
            res = not res
        j = i
    return res


def generate_coordinates(res, grid="regular"):
    """
    Generate grid coordinates for different types of grids. Currently only regular grids are implemented.

    Parameters
    ----------
    res : float
            resolution of the grid to generate

    grid : {'regular'}
            type of grid to generate. Currently only regular grids are supported

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
        lon = xarray.DataArray(np.arange(-180, 180, res), dims=("lon",), name="lon", attrs={"units": "degrees_east"})
        lat = xarray.DataArray(np.arange(-90+res/2.0, 90, res), dims=("lat",), name="lat", attrs={"units": "degrees_north"})
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

    a1 : int
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
    ens_names = ["ens", "ensemble", "member", "members"]
    for ens_name in ens_names:
        if ens_name in ds.dims:
            return True
    return False


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
        if one_var.dims[0] == "time":
            new_ds[one_name] = one_var.expand_dims("ens", 1)
        else:
            new_ds[one_name] = one_var.expand_dims("ens")
    new_ds.coords["ens"] = [member]

    # remove the ensemble_member attribute if present. it if only intented for datasets without ensemble dimension
    if "ensemble_member" in new_ds.attrs:
        del new_ds.attrs["ensemble_member"]
    return new_ds