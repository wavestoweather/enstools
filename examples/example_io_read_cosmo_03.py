#!/usr/bin/env python3
from enstools.io import read, write, drop_unused


if __name__ == "__main__":
    # read cosmo ensemble from netcdf files in forlders
    #ds = read("/project/meteo/scratch/S.Rasp/B3_collaboration/2016060600/*/lfff001?0000.nc", members_by_folder=True)
    #print(ds)

    ds = read("/project/meteo/w2w/B3/HIW_paper_BgEns/2016060600/output_cosmoDE_BgEns/*/lfff0*0.nc", members_by_folder=True)

    print("\nBefore cleanup:")
    print(ds)

    # drop unused Coordinates
    drop_unused(ds, inplace=True)

    # drop unused Variables
    ds = ds.drop(["RLON", "RLAT"])

    # show result
    print("\nAfter cleanup:")
    print(ds)

    # write result to netcdf file
    write(ds, "/local/test.nc")
