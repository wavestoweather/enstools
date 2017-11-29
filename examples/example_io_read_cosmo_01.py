#!/usr/bin/env python2
from enstools.io import read, write
import numpy as np

if __name__ == "__main__":

    # read nine ensemble members from different files
    ds1 = read("/project/meteo/work/Kevin.Bachmann/prac_predictability/data/hill_da0700_0800_40/ens00*/lfff000[1-3]0000",
              constant="/project/meteo/work/Kevin.Bachmann/prac_predictability/data/hill_da0700_0800_40/ens001/lfff00000000c",
              members_by_folder=True)
    print(ds1)

    # write to one file
    write(ds1, "/tmp/test.nc")

    # read one ensemble member for comparison
    ds2 = read("/project/meteo/work/Kevin.Bachmann/prac_predictability/data/hill_da0700_0800_40/ens007/lfff000[1-3]0000")
    print(ds2)

    # compare one variable
    print("Comparison of one member read separately to the same member from the merged read process:")
    print(np.array_equal(ds1["QV"].sel(ens=7), ds2["QV"]))
