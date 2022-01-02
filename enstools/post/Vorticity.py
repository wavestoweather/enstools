import numpy as np
import xarray
from enstools.core import check_arguments
from enstools.misc import distance
from numba import njit


@check_arguments(dims={'u': ('lat', 'lon'), 'v': ('lat', 'lon')})
def vorticity(u, v, lon, lat, fill_value=np.NaN):
    """
    Calculate vorticity and its components shear and curvature on a regular lat-lon-grid.

    Parameters
    ----------
    u: xarray.DataArray
        u-component of the wind field

    v: xarray.DataArray
        v-component of the wind field

    lon: xarray.DataArray or np.ndarray
        longitude of the input data in degrees. 1D-Array

    lat: xarray.DataArray or np.ndarray
        latitude of the input data in degrees. 1D-Array

    fill_value: float
        fill value for points at the border. Default: NaN.

    Returns
    -------
    vorticity, shear_vorticity, curve_vorticity: xarray.DataArray
    """
    # perform the actual calculation with numba
    vor, shear, curve = __vorticity(u.values, v.values, lon.values, lat.values, fill_value)

    # convert result in xarray.DataArrays
    vor = xarray.DataArray(vor, coords=[lat, lon], dims=('lat', 'lon'))
    shear = xarray.DataArray(shear, coords=[lat, lon], dims=('lat', 'lon'))
    curve = xarray.DataArray(curve, coords=[lat, lon], dims=('lat', 'lon'))
    return vor, shear, curve


@njit()
def __vorticity(u, v, lon, lat, fill_value):
    vor = np.zeros(u.shape, dtype="float")
    shear = np.zeros(u.shape, dtype="float")
    curve = np.zeros(u.shape, dtype="float")
    v1 = np.zeros(2)
    v2 = np.zeros(2)

    # loop over all grid points
    for y in range(1, u.shape[0]-1):
        for x in range(1, u.shape[1]-1):
            # any missing values?
            if (u[y,x] == fill_value or u[y,x+1] == fill_value or u[y,x-1] == fill_value
                or u[y+1,x] == fill_value or u[y-1,x] == fill_value
                or v[y,x] == fill_value or v[y,x+1] == fill_value or v[y,x-1] == fill_value
                or v[y+1,x] == fill_value or v[y-1,x] == fill_value):
                vor[y,x] = fill_value
                shear[y,x] = fill_value
                curve[y,x] = fill_value
            
            # distance on globe
            dx = distance(lat[y], lat[y], lon[x+1], lon[x-1], input_in_radian=False)
            dy = distance(lat[y-1], lat[y+1], lon[x], lon[x], input_in_radian=False)
            vor[y,x] = (v[y,x+1]-v[y,x-1]) / dx - (u[y+1,x]-u[y-1,x]) / dy

            # calculate shear-vorticity
            # calculate the wind direction
            wdir = np.arctan2(u[y,x], v[y,x])
            wspd = np.sqrt(u[y,x]**2 + v[y,x]**2)
            sin_wdir = np.sin(wdir)
            cos_wdir = np.cos(wdir)

            # calculate dot-product for four points around the reference point to the reference vector.
            # Use the wind component parallel to the reference vector from all four points and calculate
            # the vorticity based on this component, which results in the shear vorticity.
            #
            # This approach follows Berry et al. 2006.
            # Points:
            #    c
            #  b r a
            #    d

            # point a
            v1[0] = u[y,x]
            v1[1] = v[y,x]
            v2[0] = u[y,x+1]
            v2[1] = v[y,x+1]
            dp = np.dot(v1, v2) / wspd
            v_a = dp * cos_wdir

            # point b
            v2[0] = u[y,x-1]
            v2[1] = v[y,x-1]
            dp = np.dot(v1, v2) / wspd
            v_b = dp * cos_wdir

            # point c
            v2[0] = u[y+1,x]
            v2[1] = v[y+1,x]
            dp = np.dot(v1, v2) / wspd
            u_c = dp * sin_wdir

            # point d 
            v2[0] = u[y-1,x]
            v2[1] = v[y-1,x]
            dp = np.dot(v1, v2) / wspd
            u_d = dp * sin_wdir

            # calculate the shear vorticity
            shear[y,x] = (v_a-v_b) / dx - (u_c-u_d) / dy
    
    # fill values at the borders
    vor[0,:] = fill_value
    vor[-1,:] = fill_value
    vor[:,0] = fill_value
    vor[:,-1] = fill_value
    
    shear[0,:] = fill_value
    shear[-1,:] = fill_value
    shear[:,0] = fill_value
    shear[:,-1] = fill_value
    
    # calculate curvature-vorticity 
    curve = np.where((vor != fill_value) & (shear != fill_value), vor-shear, fill_value)
    return vor, shear, curve
