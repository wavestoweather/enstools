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

def test_write_file(test_dir):
    """
    create a file using enstools.io.write
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
    enstools.io.write(ds, filename)

    # read the file and check the result
    ds2 = enstools.io.read(filename)
    assert_equal(ds2["noise"].shape, (7, 5, 6))
    assert_equal(ds2["noise"].dims, ("time", "lon", "lat"))
