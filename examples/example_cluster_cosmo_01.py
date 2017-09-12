#!/usr/bin/env python2
# this example is using python 2.7 as long as eccodes is not available for python 3.x
from enstools.io import read
from enstools.cluster import prepare
from enstools.plot import contour
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# read the data
data = read("/project/meteo/w2w-db/forecasts/dwd--cosmo-de-eps--gridded--ens1-20--20140101-20161231/201407/2014072900_TOT_PREC.grib2")
print(data)

#fig, ax = contour(data["TOT_PREC"][27, 10, :, :], rotated_pole=data["rotated_pole"], coastlines="50m")
#plt.show()

# calculate clustering
labels = KMeans(n_clusters=4).fit_predict(prepare(data["TOT_PREC"][27, :, :, :]))
print(labels)
