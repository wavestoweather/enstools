#!/usr/bin/env python2
from enstools.io import read, write

if __name__ == "__main__":
    # read cosmo ensemble from netcdf files in forlders
    ds = read("/project/meteo/scratch/S.Rasp/B3_collaboration/2016060600/*/lfff001?0000.nc", members_by_folder=True)
    print(ds)
