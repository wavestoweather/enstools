import nose
import xarray
import tempfile
import numpy
import os
import shutil
import enstools

# name of the test file
test_dir = tempfile.mkdtemp()


def setup():
    """
    create a file with defined content for testing
    """
    # first file
    ds = xarray.DataArray(numpy.random.rand(7, 5, 6),
                          coords={"lon": numpy.linspace(1, 5, 5),
                                  "lat": numpy.linspace(1, 6, 6),
                                  "time": numpy.linspace(1, 7, 7)
                                  },
                          dims=["time", "lon", "lat"],
                          name="noise")
    ds.to_netcdf(os.path.join(test_dir, "01.nc"))

    # second file
    ds = xarray.DataArray(numpy.random.rand(7, 5, 6),
                          coords={"lon": numpy.linspace(1, 5, 5),
                                  "lat": numpy.linspace(1, 6, 6),
                                  "time": numpy.linspace(8, 14, 7)
                                  },
                          dims=["time", "lon", "lat"],
                          name="noise")
    ds.to_netcdf(os.path.join(test_dir, "02.nc"))


def teardown():
    """
    remove the test file created earlier
    """
    shutil.rmtree(test_dir)


def test_read_single_file():
    """
    read one netcdf file
    """
    ds = enstools.io.read(os.path.join(test_dir, "01.nc"))
    nose.tools.assert_equals(ds["noise"].shape, (7, 5, 6))
    nose.tools.assert_equals(ds["noise"].dims, ("time", "lon", "lat"))


def test_read_multiple_files():
    """
    read two netcdf files
    """
    ds = enstools.io.read([os.path.join(test_dir, "01.nc"),
                           os.path.join(test_dir, "02.nc")])
    nose.tools.assert_equals(ds["noise"].shape, (14, 5, 6))
    nose.tools.assert_equals(ds["noise"].dims, ("time", "lon", "lat"))
    numpy.testing.assert_array_equal(ds["noise"].coords["time"], numpy.linspace(1, 14, 14))


def test_open_with_wrong_argument():
    """
    try to open an something with an unsupported argument type
    """
    with numpy.testing.assert_raises(TypeError):
        ds = enstools.io.read(None)