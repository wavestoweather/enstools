import numpy as np
import xarray
from numpy.ma.core import default_fill_value
from scipy import ndimage
from enstools.core import check_arguments
from enstools.misc import count_ge
from enstools.core.parallelisation import apply_chunkwise


@check_arguments(units={"pr": "kg m-2 s-1",
                        "cape": "J kg-1",
                        "return_value": "hour"},
                 shape={"pr": "cape"})
def convective_adjustment_time_scale(pr, cape, th=1.0, fraction_above_th=0.0015):
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

    th:     float
            threshold for the precipitation rate above which the calculation should be performed [kg m-2 h-1]. Grid
            points with smaller precipitation values will contain missing values. Default: 1

    fraction_above_th: float
            fraction of grid points that must exceed the threshold defined in `th`. Default: 0.0015 (e.g. 15 of 10.000
            grid points).

    Returns
    -------
    tau:    masked_array or xarray (depending on type of pr)
            the convective adjustment time scale [h]

    Examples
    --------
    >>> cape = xarray.DataArray([500.0, 290.44], attrs={"units": "J kg-1"})
    >>> pr   = xarray.DataArray([0.0, 2.0], attrs={"units": "kg m-2 hour-1"})
    >>> np.round(convective_adjustment_time_scale(pr, cape).compute(), 4)                       # doctest:+ELLIPSIS
    <xarray.DataArray 'tauc-...' (dim_0: 2)>
    array([ nan,   1.])
    Dimensions without coordinates: dim_0

    with not enough values above the defined threshold:
    >>> np.round(convective_adjustment_time_scale(pr, cape, fraction_above_th=0.6).compute(), 4)                       # doctest:+ELLIPSIS
    <xarray.DataArray 'tauc-...' (dim_0: 2)>
    array([ nan,  nan])
    Dimensions without coordinates: dim_0

    References
    ----------
    .. [1]  Keil, C., Heinlein, F. and Craig, G. C. (2014), The convective adjustment time-scale as indicator of
            predictability of convective precipitation. Q.J.R. Meteorol. Soc., 140: 480-490. doi:10.1002/qj.2143
    """

    # TODO: tauc calculation is not chunkwise but something like layer wise
    @apply_chunkwise
    def tauc(pr, cape, th):
        # create a result array filled with the default fill value for the data type of pr
        fill_value = default_fill_value(pr)
        result = np.full_like(pr, fill_value=fill_value)

        # count values above threshold
        n_above_th = count_ge(pr, th / 3600.0)
        if n_above_th < pr.size * fraction_above_th:
            result = np.ma.masked_equal(result, fill_value)
            return result

        # Gaussian filtering
        sig = 10.  # Gaussian goes to zero 3*sig grid points from centre
        if max(pr.shape) > 3*sig:
            cape_filtered = ndimage.filters.gaussian_filter(cape, sig, mode='reflect')
            pr_filtered = ndimage.filters.gaussian_filter(pr, sig, mode='reflect')
        else:
            cape_filtered = cape
            pr_filtered = pr

        # perform the actual calculation
        ind = np.where(pr > th / 3600.0)
        result[ind] = 1.91281e-06 * cape_filtered[ind] / pr_filtered[ind]
        result = np.ma.masked_equal(result, fill_value)
        return result
    result = tauc(pr, cape, th)

    # convert the result to xarray if the input type is also xarray
    if type(pr) == xarray.DataArray:
        result = xarray.DataArray(result, coords=pr.coords, dims=pr.dims, attrs={"units": "hour"})
    return result
