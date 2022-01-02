from enstools.misc import generate_coordinates
from enstools.filters import dct_2d_regional
import xarray as xr
import numpy as np
import pytest


@pytest.fixture
def low_freq_data():
    """
    generate low frequency test data.
    """
    lon, lat = generate_coordinates(0.5, lon_range=[-20, 20], lat_range=[-10, 10])
    low_freq, _ = np.meshgrid(
        np.sin(np.linspace(-1 * np.pi, 1 * np.pi, len(lon))),
        np.zeros(40)
    )
    data = xr.DataArray(low_freq, coords=[lat, lon], dims=('lat', 'lon'), name="Test")
    return data


@pytest.fixture
def high_freq_data():
    """
    generate low frequency test data.
    """
    lon, lat = generate_coordinates(0.5, lon_range=[-20, 20], lat_range=[-10, 10])
    high_freq, _ = np.meshgrid(
        np.sin(np.linspace(-10 * np.pi, 10 * np.pi, len(lon))),
        np.zeros(40)
    )
    data = xr.DataArray(high_freq, coords=[lat, lon], dims=('lat', 'lon'), name="Test")
    return data


def test_filters_dct2d_regional_low_pass(high_freq_data, low_freq_data):
    """
    filter out the high frequency wave
    """
    # don't look at the edges. small deviations are expected.
    # high frequencies should be reduced
    filtered_low = dct_2d_regional(high_freq_data, high_freq_data['lon'], high_freq_data['lat'], low_cutoff=2000)
    np.testing.assert_array_less(filtered_low[:, 2:-2], 0.2)

    # low frequencies should not be changed significantly
    filtered_low = dct_2d_regional(low_freq_data, low_freq_data['lon'], low_freq_data['lat'], low_cutoff=2000)
    np.testing.assert_array_less(np.abs(filtered_low - low_freq_data)[:, 2:-2], 0.2)


def test_filters_dct2d_regional_high_pass(high_freq_data, low_freq_data):
    """
    filter out the low frequency wave
    """
    # don't look at the edges. small deviations are expected.
    # low frequencies should be reduced
    filtered_high = dct_2d_regional(low_freq_data, low_freq_data['lon'], low_freq_data['lat'], high_cutoff=2000)
    np.testing.assert_array_less(filtered_high[:, 2:-2], 0.2)

    # high frequencies should not be changed significantly
    filtered_high = dct_2d_regional(high_freq_data, high_freq_data['lon'], high_freq_data['lat'], high_cutoff=2000)
    np.testing.assert_array_less(np.abs(filtered_high - high_freq_data)[:, 2:-2], 0.2)
