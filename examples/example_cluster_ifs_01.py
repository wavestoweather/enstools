#!/usr/bin/env python3
from enstools.io import read
from enstools.clustering import prepare, cluster
from enstools.plot import contour, grid
import matplotlib.pyplot as plt
import argparse
import xarray

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="if provided, the plot will be saved in the given name.")
    parser.add_argument("--data", default="data", help="storage location for downloaded files.")
    args = parser.parse_args()

    # read the data
    data = read("/project/meteo/w2w/Z2/test/ifs-2016101112-*.grib")
    print(data)

    # prepare data for clustering
    z = data["z"][2, :, 0, ...] / 10
    tp = data["tp"][2, ...] * 1000
    cl_data = prepare(tp.loc[dict(lat=slice(50, 40), lon=slice(0, 5))])

    # preform the clustering
    labels = cluster("kmeans", cl_data, n_clusters=2)

    # calculate mean values for each cluster and each variable
    means_tp = []
    means_z = []
    for cl in range(labels.max() + 1):
        means_tp.append(tp[labels == cl, ...].mean("ens"))
        means_z.append(z[labels == cl, ...].mean("ens"))
    means_tp = xarray.concat(means_tp, dim="ens")
    means_z = xarray.concat(means_z, dim="ens")

    fig, _ = contour(means_tp[0], subplot_args=(2, 3, 1), colorbar=True, gridlines=True)
    fig, _ = contour(means_tp[1], subplot_args=(2, 3, 2), colorbar=True, figure=fig)
    fig, _ = contour(means_tp[1] - means_tp[0], subplot_args=(2, 3, 3), cmap="coolwarm", levels_center_on_zero=True, figure=fig)
    fig, _ = contour(means_z[0], subplot_args=(2, 3, 4), colorbar=True, figure=fig)
    fig, _ = contour(means_z[1], subplot_args=(2, 3, 5), colorbar=True, figure=fig)
    fig, _ = contour(means_z[1] - means_z[0], subplot_args=(2, 3, 6), cmap="coolwarm", levels_center_on_zero=True, figure=fig)

    if args.save is None:
        plt.show()
    else:
        fig.savefig(args.save, bbox_inches="tight", transparent=True)
