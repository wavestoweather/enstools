#!/usr/bin/env python2
from enstools.io import read
from enstools.core import init_cluster
from datetime import datetime, timedelta
import numpy as np
import os
import logging
cl = init_cluster()

# pick a random file to ensure that the cache is not involved
input_path = "/project/meteo/w2w-db/forecasts/dwd--cosmo-de-eps--gridded--ens1-20--20140101-20161231/"
first_day = datetime(2014, 1, 1)
random_day = first_day + timedelta(days=np.random.random_integers(0, 730))
input_files = os.path.join(input_path, random_day.strftime("%Y%m"), random_day.strftime("%Y%m%d") + "00_*.grib2")

logging.info("reading: %s" % input_files)
ds = read(input_files)
print(ds)