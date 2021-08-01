from enstools.post import ensemble_stat
import pytest
import xarray as xr
import numpy as np
import logging


@pytest.fixture(scope='module')
def data():
    """
    minimal test data for ensemble statistics
    """
    # create a variable with dimensions time, ens, lat, lon
    v1 = xr.DataArray(np.random.randn(2, 1000, 10, 10), dims=('time', 'ens', 'lat', 'lon'))
    return v1


def test_ensemble_stat(data):
    """
    the the ensemble stat function with minimal input
    """
    mean, min_, max_, std = ensemble_stat(data)

    # the at any given point should be 0
    np.testing.assert_array_almost_equal(mean, np.zeros_like(mean), decimal=1)

    # test for minimum and maximum and std dev
    np.testing.assert_array_equal(min_, data.values.min(axis=1))
    np.testing.assert_array_equal(max_, data.values.max(axis=1))
    np.testing.assert_array_equal(std, data.values.std(axis=1, ddof=1))


def test_ensemble_stat_ddof(data):
    """
    test if the ddof argument is used
    """
    std_ddof0 = ensemble_stat(data, stat=['std'], ddof=0)
    std_ddof1 = ensemble_stat(data, stat=['std'], ddof=1)

    # std with ddof=1 should be smaller
    np.testing.assert_array_less(std_ddof0, std_ddof1)


def test_enamble_stat_datatype(data):
    """
    make sure, that the wron data type will rais an exception
    """
    with pytest.raises(ValueError):
        _ = ensemble_stat(data.values)
