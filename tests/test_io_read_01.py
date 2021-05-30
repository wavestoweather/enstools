from numpy.testing import assert_equal
import xarray
import tempfile
import numpy
import os
import shutil
import enstools.io
import pytest


@pytest.fixture
def test_dir():
    """
    name of the test directoy
    """
    test_dir = tempfile.mkdtemp()
    yield test_dir

    # cleanup
    shutil.rmtree(test_dir)

@pytest.fixture
def file1(test_dir):
    """
    create a file with defined content for testing, netcdf
    """
    # first file
    ds = xarray.DataArray(numpy.random.rand(7, 5, 6),
                          coords={"lon": numpy.linspace(1, 5, 5),
                                  "lat": numpy.linspace(1, 6, 6),
                                  "time": numpy.linspace(1, 7, 7)
                                  },
                          dims=["time", "lon", "lat"],
                          name="noise")
    filename = os.path.join(test_dir, "01.nc")
    ds.to_netcdf(filename)
    return filename

@pytest.fixture
def file2(test_dir):
    """
    create a file with defined content for testing, HDF5
    """
    # second file
    ds = xarray.DataArray(numpy.random.rand(7, 5, 6),
                          coords={"lon": numpy.linspace(1, 5, 5),
                                  "lat": numpy.linspace(1, 6, 6),
                                  "time": numpy.linspace(8, 14, 7)
                                  },
                          dims=["time", "lon", "lat"],
                          name="noise")
    filename = os.path.join(test_dir, "02.nc")
    ds.to_netcdf(filename, engine="h5netcdf")
    return filename


def test_read_single_file(file1):
    """
    read one netcdf file
    """
    ds = enstools.io.read(file1)
    assert_equal(ds["noise"].shape, (7, 5, 6))
    assert_equal(ds["noise"].dims, ("time", "lon", "lat"))


def test_read_multiple_files(file1, file2):
    """
    read two netcdf files
    """
    ds = enstools.io.read([file1, file2])
    assert_equal(ds["noise"].shape, (14, 5, 6))
    assert_equal(ds["noise"].dims, ("time", "lon", "lat"))
    numpy.testing.assert_array_equal(ds["noise"].coords["time"], numpy.linspace(1, 14, 14))

    # same test with pattern
    ds = enstools.io.read(os.path.dirname(file1) + "/??.nc")
    assert_equal(ds["noise"].shape, (14, 5, 6))
    assert_equal(ds["noise"].dims, ("time", "lon", "lat"))
    numpy.testing.assert_array_equal(ds["noise"].coords["time"], numpy.linspace(1, 14, 14))


def test_open_with_wrong_argument():
    """
    try to open an something with an unsupported argument type
    """
    with numpy.testing.assert_raises(NotImplementedError):
        ds = enstools.io.read(None)
