#!/usr/bin/env python2
# this example is using python 2.7 as long as eccodes is not available for python 3.x
from enstools.io import read
from enstools.cluster import prepare, cluster
from enstools.plot import contour, grid
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="if provided, the plot will be saved in the given name.")
    parser.add_argument("--data", default="data", help="storage location for downloaded files.")
    args = parser.parse_args()

    # read the data
    data = read("/project/meteo/w2w-db/forecasts/dwd--cosmo-de-eps--gridded--ens1-20--20140101-20161231/201407/2014072900_*.grib2")

    # calculate clustering
    # --------------------
    # 1. prepare the data (reshape, normalize, etc...), use the last four time steps
    cl_data = prepare(data["TOT_PREC"][24:28, :, :, :], data["CAPE_ML"][24:28, :, :, :])
    # 2. perform the actual clustering, the number of clusters is estimated automatically
    labels = cluster("kmeans", cl_data, n_clusters=4)
    print("Clustering: %s" % ", ".join(map(lambda x:str(x), labels)))

    # plot the whole ensemble
    # -----------------------
    # use a different color map for each cluster
    cmap_names = ["Purples", "Blues", "Greens", "Oranges", "Reds", "Greys"]
    cmaps = list(map(lambda x: cmap_names[x], labels))

    # specifying rlon and rlat is not necessary, but using the rotated coordinates is much faster compared to the 2d
    # coordinate arrays.
    fig, ax = grid(contour,
                   data["TOT_PREC"][27, :, :, :],
                   data["rlon"],
                   data["rlat"],
                   shape=(4, 5), cmaps=cmaps, rotated_pole=data["rotated_pole"], colorbar=False)
    if args.save is None:
        plt.show()
    else:
        fig.savefig(args.save, bbox_inches="tight", transparent=True)
