#!/usr/bin/env python2
# this example is using python 2.7 as long as eccodes is not available for python 3.x
from enstools.io import read
from enstools.cluster import prepare, cluster
from enstools.plot import contour
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# read the data
data = read("/project/meteo/w2w-db/forecasts/dwd--cosmo-de-eps--gridded--ens1-20--20140101-20161231/201407/2014072900_*.grib2")
print(data)

#fig, ax = contour(data["TOT_PREC"][27, 10, :, :], rotated_pole=data["rotated_pole"], coastlines="50m")
#plt.show()


# calculate clustering
# --------------------
# 1. prepare the data (reshape, normalize, etc...)
cl_data = prepare(data["TOT_PREC"][24:28, :, :, :], data["CAPE_ML"][24:28, :, :, :], ens_dim=1)

# 2. perform the actual clustering
labels = cluster("kmeans", cl_data)

print(labels)
