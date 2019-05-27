#!/usr/bin/env python3
from enstools.io import read, write

if __name__ == "__main__":
    # read cosmo ensemble with one member per file
    ds = read("/project/meteo/w2w/Unwetter2016/deeps/deeps_2016060600_m*.grib2")
    print(ds)