from enstools.post import vorticity
from enstools.misc import generate_coordinates
import numpy as np
import xarray as xr


def test_vorticity():
    """
    test vorticity calculation with a test case that should have zero vorticity
    """
    # generate test data for "Irrotational vortex"
    # the actual center of the vortex is intentionally not part of the domain.
    lon, lat = generate_coordinates(0.5, lon_range=[-5, 5], lat_range=[2, 10])
    u_no_vort = np.empty((len(lat), len(lon)))
    v_no_vort = np.empty((len(lat), len(lon)))
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            u_no_vort[ilat, ilon] = -lat[ilat]
            v_no_vort[ilat, ilon] = lon[ilon]
            speed = np.sqrt(u_no_vort[ilat, ilon]**2 + v_no_vort[ilat, ilon]**2)
            u_no_vort[ilat, ilon] = u_no_vort[ilat, ilon] / speed**2 * 10
            v_no_vort[ilat, ilon] = v_no_vort[ilat, ilon] / speed**2 * 10
    u_no_vort = xr.DataArray(u_no_vort, coords=[lat, lon], dims=("lat", "lon"))
    v_no_vort = xr.DataArray(v_no_vort, coords=[lat, lon], dims=("lat", "lon"))

    # calculate vorticity
    vor, shear, curve = vorticity(u_no_vort, v_no_vort, lon, lat)
    # the vorticity itself is supposed to by about zero
    np.testing.assert_array_almost_equal(vor[1:-1,1:-1], 0)

    # curvature vorticity should be larger zero
    np.testing.assert_array_less(0.5e-6, curve[1:-1,1:-1])

    # shear vorticity should be smaller then zero
    np.testing.assert_array_less(shear[1:-1,1:-1], -0.5e-6)

    # outermost points should be all NaN
    for array in [vor, shear, curve]:
        assert np.all(np.isnan(array[0, :]))
        assert np.all(np.isnan(array[-1, :]))
        assert np.all(np.isnan(array[:, 0]))
        assert np.all(np.isnan(array[:, -1]))
