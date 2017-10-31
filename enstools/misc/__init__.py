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