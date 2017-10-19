import os
import logging
import bz2
import six
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