#!/usr/bin/env python3
from enstools.io import read
from enstools.core import init_cluster, set_behavior
from time import sleep
cl = init_cluster()

set_behavior(log_level="DEBUG")

ds = read("/project/meteo/w2w-db/forecasts/ecmwf--oper-eps--gridded--ens1-51--20160919-20161102/reduced/20161011_00_ecmwf_ensemble_forecast.MODEL_LEVELS.NAWDEX_LLML.00?.ml.grb", in_memory=True)
#ds = read("/project/meteo/w2w-db/forecasts/ecmwf--oper-eps--gridded--ens1-51--20160919-20161102/netcdf/20160922_00_ecmwf_ensemble_forecast.MODEL_LEVELS.NAWDEX_LLML.00?.ml.nc", in_memory=True)
print(ds)

sleep(30)

del ds

sleep(5)