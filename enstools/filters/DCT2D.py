from enstools.core import check_arguments
from enstools.misc import distance
import numpy as np
from scipy.fft import dctn, idctn
from numba import njit, objmode


@check_arguments(dims={'variable': ('lat', 'lon')})
def dct_2d_regional(variable, lon, lat, high_cutoff=0.0, low_cutoff=0.0):
    """
    Filter an input array on a regular lat-lon grid using a discrete cosine transformation

    Parameters
    ----------
    variable: xarray.DataArray
        2d variable on a regular lat-lon grid

    lon: xarray.DataArray
        longitude of input data, 1D.

    lat: xarray.DataArray
        latitude of input data, 1D.

    low_cutoff: float
        wave length for low-pass filter in km. Shorter wave lengths are removed. The default
        value of zero means, that no filter is applied.

    high_cutoff: float
        wave length for high-pass filter in km. Longer wave lengths are removed. The default
        value of zero means, that no filter is applied.

    Returns
    -------
    filtered: xarray.DataArray
        filtered version of the input
    """

    # forward pass of DCT on the input data
    coefficients = dctn(variable.values, type=2)

    # apply the actual filter
    __dct_2d_regional(coefficients, lon.values, lat.values, high_cutoff, low_cutoff)

    # perform the inverse transformation
    filtered = idctn(coefficients, type=2)

    # pack result into xarray object
    result = variable.copy(data=filtered)
    return result


@njit()
def __dct_2d_regional(transformed_array, lon, lat, high_cutoff=0.0, low_cutoff=0.0):

    # low-pass filter
    if low_cutoff > 0:
        # calculate the wave number corresponding to the cut-off wave length
        wn_lon, wn_lat = get_wavenumber_from_length(low_cutoff, lon, lat)

        # use an elliptic mask to filter the coefficients (compare Denis et al. 2002)
        # set all the coefficients outside the ellipse to 0
        for i in range(len(lat)):
            for j in range(len(lon)):
                # No idea why the Python and Fortran wn_pos different in the factor 2 in the end here:
                wn_pos = np.sqrt(float(j)**2 / (wn_lon-1)**2 + float(i)**2 / (wn_lat-1)**2)  
                if wn_pos > 1.:
                    transformed_array[i,j] = 0.

    # high-pass filter
    if high_cutoff > 0:
        # calculate the wave number corresponding to the cut-off wave length
        wn_lon, wn_lat = get_wavenumber_from_length(high_cutoff, lon, lat)

        # use an elliptic mask to filter the coefficients (compare Denis et al. 2002)
        # set all the coefficients inside the ellipse to 0
        for i in range(len(lat)):
            for j in range(len(lon)):
                # No idea why the Python and Fortran wn_pos different in the factor 2 in the end here:
                wn_pos = np.sqrt(float(j)**2 / (wn_lon-1)**2 + float(i)**2 / (wn_lat-1)**2)
                if wn_pos <= 1.:
                    transformed_array[i,j] = 0.
    return transformed_array


@njit()
def get_wavenumber_from_length(cutoff_length, coord_lon, coord_lat):
    """
    Find the wave number that corresponds to a given length in lon- and lat-direction.
    """

    # calculate the lat-extent of the array along an axis that is centered in the domain
    length = 0.
    x = int(len(coord_lon) / 2) # maybe +1 
    for i in range(len(coord_lat)-1):
        length += distance(coord_lat[i], coord_lat[i+1], coord_lon[x], coord_lon[x], input_in_radian=False)
    wave_number_lat = np.rint(length / 1000.0 / cutoff_length * 2) + 1

    # calculate the lon-extent of the array along an axis that is centered in the domain
    length = 0.
    y = int(len(coord_lat) / 2) # maybe +1
    for i in range(len(coord_lon)-1):
        length += distance(coord_lat[y], coord_lat[y], coord_lon[i], coord_lon[i+1], input_in_radian=False)
    wave_number_lon = np.rint(length / 1000.0 / cutoff_length * 2) + 1

    # limit the wave numbers
    if wave_number_lon < 2:
        wave_number_lon = 2
    if wave_number_lat < 2:
        wave_number_lat = 2
    return wave_number_lon, wave_number_lat
