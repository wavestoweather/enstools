import numpy as np
import xarray
from numpy.ma.core import default_fill_value
from scipy import ndimage
from enstools.core import check_arguments
from itertools import product


@check_arguments(units={"pr": "kg m-2 s-1",
                        "cape": "J kg-1",
                        "return_value": "hour"},
                 shape={"pr": "cape"})
def convective_adjustment_time_scale(pr, cape, th=1.0):
    """
    Calculate the convective adjustment time scale from precipitation and CAPE as described in [1]_. A gaussian filter
    is applied to the input data if at least one dimension has more then 30 grid points.

    Parameters
    ----------
    pr:     xarray.DataArray
            Hourly or longer accumulated precipitation converted to [kg m-2 s-1].

    cape:   xarray.DataArray
            The CAPE of mean surface layer parcel [J kg-1].
            For example, mean of start and end value of the period of the accumulation.

    th:     scalar
            threshold for the precipitation rate above which the calculation should be performed [kg m-2 h-1]. Grid
            points with smaller precipitation values will contain missing values. Default: 1

    Returns
    -------
    tau:    masked_array or xarray (depending on type of pr)
            the convective adjustment time scale [h]

    Examples
    --------
    >>> cape = xarray.DataArray([500.0, 290.44], attrs={"units": "J kg-1"})
    >>> pr   = xarray.DataArray([0.0, 2.0], attrs={"units": "kg m-2 hour-1"})
    >>> np.round(convective_adjustment_time_scale(pr, cape), 4)
    <xarray.DataArray (dim_0: 2)>
    array([nan,  1.])
    Dimensions without coordinates: dim_0

    References
    ----------
    .. [1]  Keil, C., Heinlein, F. and Craig, G. C. (2014), The convective adjustment time-scale as indicator of
            predictability of convective precipitation. Q.J.R. Meteorol. Soc., 140: 480-490. doi:10.1002/qj.2143
    """

    # Gaussian filtering
    sig = 10.  # Gaussian goes to zero 3*sig grid points from centre
    if max(pr.shape) > 3*sig:
        if len(pr.shape) <= 2:
            cape_filtered = ndimage.filters.gaussian_filter(cape, sig, mode='reflect')
            pr_filtered = ndimage.filters.gaussian_filter(pr, sig, mode='reflect')
        else:
            cape_filtered = np.empty_like(cape)
            pr_filtered = np.empty_like(pr)
            for x in product(*map(lambda x:range(x),cape.shape[:-2])):
                idx = x+(slice(None,None),slice(None,None))
                cape_filtered[idx] = ndimage.filters.gaussian_filter(cape[idx], sig, mode='reflect')
                pr_filtered[idx] = ndimage.filters.gaussian_filter(pr[idx], sig, mode='reflect')
    else:
        cape_filtered = cape
        pr_filtered = pr

    # create a result array filled with the default fill value for the data type of pr
    fill_value = default_fill_value(pr)
    result = np.full_like(pr, fill_value=fill_value)

    # perform the actual calculation
    ind = np.where(pr_filtered > th / 3600.0)
    result[ind] = 1.91281e-06 * cape_filtered[ind] / pr_filtered[ind]
    result = np.ma.masked_equal(result, fill_value)

    # convert the result to xarray if the input type is also xarray
    if type(pr) == xarray.DataArray:
        result = xarray.DataArray(result, coords=pr.coords, dims=pr.dims, attrs={"units": "hour"})
    return result
