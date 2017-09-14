"""
different methods to display ensemble data
"""
import numpy as np
import xarray
from enstools.plot.core import get_nice_levels, get_coordinates_from_xarray


def grid(plot_function, variable, lon=None, lat=None, figure=None, axes=None, shape=None, cmaps=None, **kwargs):
    """
    Create a multi-panel plot by mapping the first dimension of the input array to a plot function specified in the
    second argument.

    Parameters
    ----------
    plot_function : Callable
            the plot function to use. supported are all function in enstools.plot

    variable : xarray.DataArray or np.ndarray
            the data to plot. Should be an 3d-Array, the first dimension is used to create the grid.

    lon : xarray.DataArray or np.ndarray or str
            longitude coordinate or name of longitude coordinate. The name may only be used for xarray variables.

    lat : xarray.DataArray or np.ndarray or str
            latitude coordinate or name of longitude coordinate. The name may only be used for xarray variables.

    figure : matplotlib.figure.Figure
            If provided, this figure instance will be used (and returned), otherwise a new
            figure will be created.

    axes : array of matplotlib.axes.Axes
            if provided, this has to be an array of matplotlib.axes.Axes instances with the desired shape of the plot.

    shape : tuple
            The shape of the multi-panel plot (nrows, ncols). Not providing shape or axes is an error!

    cmaps : list
            A list of cmap names. The number of entries has to be the number of panel plots to create.

    **kwargs:
            Keyword arguments, they are forwarded top the actual plt function that creates the individual panels

    Returns
    -------
    tuple:
            fig, ax, where ax is an array for Axes objects

    Examples
    --------

    >>> fig, ax = grid(contour, data["TOT_PREC"][27, :, :, :], data["rlon"], data["rlat"], shape=(4, 5), cmaps=cmaps, rotated_pole=data["rotated_pole"], colorbar=False)    # doctest: +SKIP

    .. figure:: images/example_cluster_cosmo_01.png

        Example of displaying the result of K-Mean clustering for the COSMO-DE Ensemble.
        Each cluster has a different colormap.

    """
    # check arguments
    if variable.ndim != 3:
        raise ValueError("the input variable has to have three dimension, given: %d" % variable.ndim)
    if shape is None and axes is not None:
        shape = axes.shape
    if shape is None:
        raise ValueError("either shape or axes has to be provided!")
    if len(shape) != 2:
        raise ValueError("shape has to be 2d!")

    # create an array for the axes objects
    if axes is None:
        axes = np.empty(shape, dtype=object)

    # calculate levels for all panels
    if kwargs.get("levels", None) is None:
        kwargs["levels"] = get_nice_levels(variable)

    # assign color maps
    cmap = kwargs.get("cmap", None)
    if cmaps is None:
        cmaps = [cmap] * variable.shape[0]
    if len(cmaps) != variable.shape[0]:
        raise ValueError("wrong number of colormaps given!")

    # loop over all panels
    for ipanel in range(np.product(shape)):
        figure, axes[np.unravel_index(ipanel, shape)] = plot_function(variable[ipanel, ...],
                                                                      lon,
                                                                      lat,
                                                                      figure=figure,
                                                                      subplot_args=(shape[0], shape[1], ipanel+1),
                                                                      cmap=cmaps[ipanel],
                                                                      **kwargs)
    return figure, axes
