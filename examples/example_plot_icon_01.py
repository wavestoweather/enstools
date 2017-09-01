#!/usr/bin/env python2
# this example is using python 2.7 as long as eccodes is not available for python 3.x
import enstools.io
import urllib
import os
import bz2
import logging
from datetime import datetime


def download(url, destination):
    """
    Download a file from the web

    Parameters
    ----------
    url : string
            the web address

    destination : string
            path to store the file. The file will only be downloaded once. If the name ends with bz2, it will
            automatically be uncompressed
    """
    if destination.endswith(".bz2"):
        destination_intern = destination[:-4]
    else:
        destination_intern = destination

    if os.path.exists(destination_intern):
        logging.warning("file not downloaded because it is already present: %s" % url)
        return

    # download
    fn, hd = urllib.urlretrieve(url, destination)

    # uncompress
    if destination.endswith(".bz2"):
        # uncompress
        bfile = bz2.BZ2File(destination)
        dfile = open(destination_intern, "wb")
        dfile.write(bfile.read())
        dfile.close()
        bfile.close()
        # delete compressed
        os.remove(destination)


if __name__ == "__main__":
    # ensure that the data directory is available
    if not os.path.exists("data"):
        os.makedirs("data")
    # download the 24h forecast from today 00 UTC.
    today = datetime.now().date()
    tp_file = "ICON_EU_single_level_elements_TOT_PREC_%s00_024.grib2" % today.strftime("%Y%m%d")
    # download the data itself
    download("http://opendata.dwd.de/weather/icon/eu_nest/grib/tot_prec/%s.bz2" % tp_file, "data/%s.bz2" % tp_file)

    # read the grib file
    grib = enstools.io.read("data/%s" % tp_file)

    # write to netcdf file
    enstools.io.write(grib, "data/%s" % tp_file.replace(".grib2", ".nc"))

