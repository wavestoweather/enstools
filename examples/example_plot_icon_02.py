#!/usr/bin/env python3
import os
import enstools.io
import enstools.plot
import matplotlib.pyplot as plt
from enstools.misc import download
from datetime import datetime
import argparse


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
    for variable_name in ["PMSL", "TOT_PREC"]:
        file_name = "icon_global_icosahedral_single-level_%s00_024_%s.grib2" % (today.strftime("%Y%m%d"), variable_name)
        download("https://opendata.dwd.de/weather/nwp/icon/grib/00/%s/%s.bz2" % (variable_name.lower(), file_name),
             "%s/%s.bz2" % (args.data, file_name))
        data_files.append("%s/%s" % (args.data, file_name))

    # download the grid definition file
    download("http://icon-downloads.zmaw.de/grids/public/edzw/icon_grid_0026_R03B07_G.nc",
             "%s/icon_grid_0026_R03B07_G.nc" % args.data)
    data_files.append("%s/icon_grid_0026_R03B07_G.nc" % args.data)

    # read the grib files
    data = enstools.io.read(data_files, merge_same_size_dim=True)

    # variable names depend on the presence of the DWD grib definitions
    if "PMSL" in data:
        PMSL = "PMSL"
        TOT_PREC = "TOT_PREC"
    else:
        PMSL = "prmsl"
        TOT_PREC = "tp"

    # create a basic map plot
    fig, ax1 = enstools.plot.contour(data[PMSL][0, :] / 100.0, gridlines=True, subplot_args=(121,))
    fig, ax2 = enstools.plot.contour(data[TOT_PREC][0, :], figure=fig, subplot_args=(122,))
    if args.save is None:
        plt.show()
    else:
        fig.savefig(args.save, bbox_inches="tight", transparent=True)
