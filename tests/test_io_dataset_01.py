# Tests for functions from enstools.io.dataset
import xarray
import numpy
from enstools.io import drop_unused


def test_drop_unused_coords():
    """
    test for drop_unused from enstools.io
    """
    # create a test dataset
    array = xarray.DataArray(numpy.random.rand(7, 5, 6),
                          coords={"lon": numpy.linspace(1, 5, 5),
                                  "lat": numpy.linspace(1, 6, 6),
                                  "time": numpy.linspace(1, 7, 7)
                                  },
                          dims=["time", "lon", "lat"],
                          name="noise")
    # 1d-coordinate with same name as dimension
    unused_coord1 = xarray.DataArray(numpy.linspace(1, 5, 5),
                          coords={"rlon": numpy.linspace(1, 5, 5)},
                          dims=["rlon"],
                          name="rlon")
    # 2d-coordinate with different name than dimensions
    unused_coord2 = xarray.DataArray(numpy.random.rand(5, 6),
                          coords={"rlon": numpy.linspace(1, 5, 5),
                                  "rlat": numpy.linspace(1, 6, 6)},
                          dims=["rlon", "rlat"],
                          name="srlat")
    # create the dataset
    ds = xarray.Dataset({"noise":array, "rlon": unused_coord1, "srlat": unused_coord2})
    # mark the 2d-coord as coord
    ds = ds.set_coords("srlat")

    # drop the unused coordinates
    ds = drop_unused(ds)

    # check absence of unused coords and dims
    numpy.testing.assert_equal("srlat" in ds.coords, False)
    numpy.testing.assert_equal("srlat" in ds.variables, False)
    numpy.testing.assert_equal("rlon" in ds.coords, False)
    numpy.testing.assert_equal("rlon" in ds.variables, False)
    numpy.testing.assert_equal("rlon" in ds.dims, False)
    numpy.testing.assert_equal("rlat" in ds.dims, False)

