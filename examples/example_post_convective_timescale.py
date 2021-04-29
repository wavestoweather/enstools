#!/usr/bin/env python3
import os
import enstools.io
import enstools.plot
from enstools.opendata import retrieve_nwp
from enstools.post.ConvectiveAdjustmentTimescale import convective_adjustment_time_scale
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
import pandas


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="If provided, the plot will be saved in the given name.")
    parser.add_argument("--data", default="data", help="Storage location for downloaded files.")
    parser.add_argument("--input", help="Optional input file. If present, it is used instead of downloading todays forecast. TOT_PREC, CAPE_ML, and coordinates are expected.")
    args = parser.parse_args()

    # ensure that the data directory is available
    if not os.path.exists(args.data):
        os.makedirs(args.data)

    # download forecast from today 00 UTC.
    if args.input is not None:
        grib_file = args.input
    else:
        grib_file = retrieve_nwp(model="icon-d2", 
                                grid_type="regular-lat-lon", 
                                variable=["cape_ml", "tot_prec"], 
                                forecast_hour=list(np.arange(25, dtype=np.int64)),
                                init_time=0,
                                dest=args.data,
                                merge_files=True)

    # read the grib file
    grib = enstools.io.read(grib_file)

    # variables names (depending on availability of DWD grib definitions)
    tp_name = "tp" if not "TOT_PREC" in grib else "TOT_PREC" 

    # calculate 3-hourly precipitation from accumulated values
    grib3h = grib.isel(time=slice(None, None, 12))
    tp3h = (grib3h[tp_name][1:,...] - np.array(grib3h[tp_name][0:-1,...])) / 3.0
    tp3h.attrs["units"] = "kg m-2 hour-1"
 
    # calculate 3-hourly mean values of cape, ignore last 45min
    cape3h = grib["CAPE_ML"].resample(time="3H", label='right').reduce(np.mean)[0:-1,...]
    cape3h.attrs["units"] = "J/kg"
 
    # calculate convective adjustment timescale with default parameters
    tau = convective_adjustment_time_scale(pr=tp3h, cape=cape3h)

    # create a basic contour plot and show or save it
    hour_to_plot=4
    fig, ax1 = enstools.plot.contour(tp3h[hour_to_plot,...], coastlines="50m", subplot_args=(221,), subplot_kwargs={'title': "TOT_PREC [mm/h]"})
    fig, ax2 = enstools.plot.contour(cape3h[hour_to_plot,...], coastlines="50m", figure=fig, subplot_args=(222,), subplot_kwargs={'title': "CAPE_ML [J/kg]"})
    fig, ax3 = enstools.plot.contour(tau[hour_to_plot,...], coastlines="50m", figure=fig, subplot_args=(223,), subplot_kwargs={'title': "TAU [hour]"})

    # add interval as title
    interval_start = pandas.to_datetime(tp3h["time"][hour_to_plot-1].values)
    interval_end = pandas.to_datetime(tp3h["time"][hour_to_plot].values)
    fig.suptitle(f'{interval_start.strftime("%Y-%m-%d %H:%M")} - {interval_end.strftime("%H:%M")}')
    if args.save is None:
        plt.show()
    else:
        fig.savefig(args.save, bbox_inches="tight", transparent=False, dpi=200)
