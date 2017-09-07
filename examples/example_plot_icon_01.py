#!/usr/bin/env python2
# this example is using python 2.7 as long as eccodes is not available for python 3.x
import os
import enstools.io
import enstools.plot
import matplotlib.pyplot as plt
from enstools.misc import download
from datetime import datetime


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

    # create a basic map plot
    ax = enstools.plot.map_plot(grib["TOT_PREC"][0, :, :])
    plt.show()
