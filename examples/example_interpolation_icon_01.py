#!/usr/bin/env python2
# this example is using python 2.7 as long as eccodes is not available for python 3.x
import os
import numpy as np
import enstools.io
import enstools.plot
import matplotlib.pyplot as plt
from enstools.misc import download
from enstools.interpolation import nearest_neighbour
from datetime import datetime
import argparse
import cartopy.crs as ccrs


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="if provided, the plot will be saved in the given name.")
    parser.add_argument("--data", default="data", help="storage location for downloaded files.")
    args = parser.parse_args()

    # ensure that the data directory is available
    if not os.path.exists(args.data):
        os.makedirs(args.data)

    # download the 24h forecast from today 00 UTC.
    today = datetime.now().date()
    data_files = []
    for variable_name in ["FI"]:
        file_name = "ICON_iko_pressure_level_elements_world_%s_%s00_024.grib2" % (variable_name, today.strftime("%Y%m%d"))
        download("http://opendata.dwd.de/weather/icon/global/grib/00/%s/%s.bz2" % (variable_name.lower(), file_name),
             "%s/%s.bz2" % (args.data, file_name))
        data_files.append("%s/%s" % (args.data, file_name))

    # download the grid definition file
    download("http://opendata.dwd.de/weather/lib/icon_grid_0026_R03B07_G.nc.bz2",
             "%s/icon_grid_0026_R03B07_G.nc.bz2" % args.data)
    data_files.append("%s/icon_grid_0026_R03B07_G.nc" % args.data)

    # read the grib files
    data = enstools.io.read(data_files)

    # interpolate onto a regular grid
    lon = np.arange(-180, 180, 0.25)
    lat = np.arange(-89, 89.25, 0.25)
    interpol = nearest_neighbour(np.rad2deg(data["clon"]), np.rad2deg(data["clat"]), lon, lat, input_grid="unstructured", output_grid="regular")
    gridded = interpol(data["FI"])
    gridded.to_netcdf("test.nc")
    print(data["clon"])

    fig, ax1 = enstools.plot.contour(data["FI"][0, 1, ...], gridlines=True, subplot_args=(121,))
    fig, ax2 = enstools.plot.contour(gridded[0, 1, ...], figure=fig, subplot_args=(122,))

    if args.save is None:
        plt.show()
    else:
        fig.savefig(args.save, bbox_inches="tight", transparent=True)
