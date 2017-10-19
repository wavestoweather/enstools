#!/usr/bin/env python3
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from enstools.io import read
from enstools.misc import download
from enstools.post import rossby_wave_packets_diag, rossby_wave_packets_plot

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="if provided, the plot will be saved in the given name.")
    parser.add_argument("--data", default="data", help="storage location for downloaded files.")
    args = parser.parse_args()

    # ensure that the data directory is available
    if not os.path.exists(args.data):
        os.makedirs(args.data)

    # download the example data
    download("https://syncandshare.lrz.de/dl/fiTi9jXWQAuk28ugV4GpZbEB/example_post_rwp_01.nc", "%s/example_post_rwp_01.nc" % args.data)

    # read the example data
    nc = read("%s/%s" % (args.data, "example_post_rwp_01.nc"))

    # calculate the diagnostic for one point in time
    rwp = rossby_wave_packets_diag(nc["u"], nc["v"], nc["z"], date=datetime(2002, 8, 7))

    # create standard plots
    fig, ax = rossby_wave_packets_plot(rwp)
    if args.save is None:
        plt.show()
    else:
        fig.savefig(args.save, bbox_inches="tight", transparent=True)
