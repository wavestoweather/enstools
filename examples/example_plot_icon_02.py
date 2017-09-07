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
    variable_name = "PMSL"
    today = datetime.now().date()
    tp_file = "ICON_iko_single_level_elements_world_%s_%s00_024.grib2" % (variable_name, today.strftime("%Y%m%d"))
    # download the data itself
    download("http://opendata.dwd.de/weather/icon/global/grib/%s/%s.bz2" % (variable_name.lower(), tp_file),
             "data/%s.bz2" % tp_file)

    # download the grid definition file
    download("http://opendata.dwd.de/weather/lib/icon_grid_0026_R03B07_G.nc.bz2",
             "data/icon_grid_0026_R03B07_G.nc.bz2")

    # read the grib file
    grib = enstools.io.read(["data/%s" % tp_file, "data/icon_grid_0026_R03B07_G.nc"], merge_same_size_dim=True)
    print(grib)

    # create a basic map plot
    ax = enstools.plot.map_plot(grib[variable_name][0, :], gridlines=True)
    plt.show()
