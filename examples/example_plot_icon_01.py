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
    tp_file = "icon-eu_europe_regular-lat-lon_single-level_%s00_024_TOT_PREC.grib2" % today.strftime("%Y%m%d")
    download("http://opendata.dwd.de/weather/nwp/icon-eu/grib/00/tot_prec/%s.bz2" % tp_file,
             "%s/%s.bz2" % (args.data, tp_file))

    # read the grib file
    grib = enstools.io.read("%s/%s" % (args.data, tp_file))

    # create a basic contour plot and show or save it
    fig, ax = enstools.plot.contour(grib["TOT_PREC"][0, :, :], coastlines="50m")
    if args.save is None:
        plt.show()
    else:
        fig.savefig(args.save, bbox_inches="tight", transparent=True)
