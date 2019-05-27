#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
from enstools.misc import download, generate_coordinates
from enstools.interpolation import nearest_neighbour
from enstools.io import read
from enstools.plot import contour
from datetime import datetime, timedelta
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
    download("http://icon-downloads.zmaw.de/grids/public/edzw/icon_grid_0026_R03B07_G.nc",
             "%s/icon_grid_0026_R03B07_G.nc" % args.data)
    data_files.append("%s/icon_grid_0026_R03B07_G.nc" % args.data)

    # read the grib files
    data = read(data_files)
    phi = data["FI"].sel(isobaricInhPa=300, time=today + timedelta(days=1))

    # interpolate onto a regular grid
    lon, lat = generate_coordinates(0.25)
    interpol = nearest_neighbour(data["clon"], data["clat"], lon, lat, src_grid="unstructured", dst_grid="regular")
    phi_rg = interpol(phi)
    print(phi_rg)

    # plot both fields for comparison
    fig, ax1 = contour(phi, gridlines=True, subplot_args=(121,), projection=ccrs.Robinson())
    fig, ax2 = contour(phi_rg, figure=fig, subplot_args=(122,), projection=ccrs.Robinson())

    if args.save is None:
        plt.show()
    else:
        fig.savefig(args.save, bbox_inches="tight", transparent=True)
