#!/usr/bin/env python2
from enstools.io import read
from enstools.core import init_cluster
from enstools.post import convective_adjustment_time_scale
from datetime import datetime, timedelta
import numpy as np
import os
import logging
import time
cl = init_cluster()

# pick a random file to ensure that the cache is not involved
input_path = "/project/meteo/w2w-db/forecasts/dwd--cosmo-de-eps--gridded--ens1-20--20140101-20161231/"
first_day = datetime(2014, 1, 1)
random_day = first_day + timedelta(days=np.random.random_integers(0, 730))
input_files = os.path.join(input_path, random_day.strftime("%Y%m"), random_day.strftime("%Y%m%d") + "00_*.grib2")

logging.info("reading: %s" % input_files)
ds = read(input_files, in_memory=True)
print(ds)
#print(ds2)
#time.sleep(10)

# small computation
ds["TOT_PREC"].attrs["units"] = "kg m-2 / 3600 / s"
ds["CAPE_ML"].attrs["units"] = "J kg-1"
#tauc = convective_adjustment_time_scale(ds["TOT_PREC"][17,0,...] - ds["TOT_PREC"][16,0,...], ds["CAPE_ML"][17,0,...])
tauc = convective_adjustment_time_scale(ds["TOT_PREC"], ds["CAPE_ML"])
print(tauc)
print(tauc.compute())

del ds
del tauc

