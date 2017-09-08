"""
Basic plot routines usable to construct more advanced plots
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.axes import Axes
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_size import AxesY, Fraction
from numba import jit
from multipledispatch import dispatch
from enstools.misc import point_in_polygon
import xarray
import six
import math


# names for coordinates
__lon_names = ["lon", "lons", "longitude", "longitudes", "clon", "rlon"]
__lat_names = ["lat", "lats", "latitude", "latitudes", "clat", "rlat"]

# cache for expensively calculated triangulations
__triangulation_cache = {}


def __is_global(lon, lat):
    """
    check if coordinates belong to a global dataset

    Parameters
    ----------
    lon : np.ndarray or xarray.DataArray
    lat : np.ndarray or xarray.DataArray

    Returns
    -------
    bool
    """
    if lon.max() - lon.min() > 350 and lat.max() - lat.min() > 80:
        return True
    return False


def __get_order_of_magnitude(value):
    """
    get the order of magnitude for a given value. This function may be used to create nice
    colorbar levels.

    Parameters
    ----------
    value : float

    Returns
    -------
    float

    Examples
    --------
    >>> __get_order_of_magnitude(3)
    5
    >>> __get_order_of_magnitude(7)
    10
    >>> __get_order_of_magnitude(0.7)
    1
    """
    x1 = [1, 2, 5]
    x2 = -10
    i = 0
    while True:
        x3 = x1[i] * 10**x2
        if x3 >= abs(value):
            break
        if i < 2:
            i += 1
        else:
            i = 0
            x2 = x2 + 1
    if value < 0:
        x3 = x3 * -1
    return x3


def __get_nice_levels(variable, nlevel=11, min_percentile=0.1, max_percentile=99.9):
    """
    Create levels for contour plots

    Parameters
    ----------
    variable : xarray.DataArray or np.ndarray
            the values for which the levels should be created

    nlevel : int
            number of levels

    max_percentile : float
            plots often look nicer if not the absolute maximum but is
            used, but a percentile.

    Returns
    -------
    np.ndarray
            level-array usable in matplotlib plotting methods
    """
    # use a percentile instead of the actual maximum
    minval, maxval = np.percentile(variable, [min_percentile, max_percentile])
    perc10 = __get_order_of_magnitude(maxval - minval) / 10.0

    # adjust minimum, especially around zero
    minval_is_adjusted = False
    if minval > 0 > minval - perc10:
        minval = 0
        minval_is_adjusted = True
    elif minval < 0 < maxval and abs(minval) < abs(maxval) / 1000:
        minval = 0
        minval_is_adjusted = True
    else:
        minval -= np.mod(minval, perc10)

    # adjust maximum
    maxval = maxval + perc10 - np.mod(maxval, perc10)
    step = __get_order_of_magnitude((maxval - minval) / nlevel)
    if not minval_is_adjusted and min_percentile > 0.0 and minval > variable.min():
        minval -= step
    # modify the maximum to ensure, that it is an integer multiple of the step
    range = maxval - minval
    if np.mod(range, step) != 0:
        maxval += step - np.mod(range, step)
    return np.linspace(minval, maxval, nlevel)


@jit(nopython=True)
def __mask_triangles(triangles, proj_coords, boundary, mask):
    # coordinates are given as 2d-arrays
    x_coords_proj = proj_coords[:, 0]
    y_coords_proj = proj_coords[:, 1]

    # shrink the boundary a bit to be sure that no point is on the boundary
    boundary_x = np.empty(boundary.shape[0])
    boundary_y = np.empty(boundary.shape[0])
    offset = boundary.max() * 0.002
    for i in range(boundary.shape[0]):
        a = math.atan2(boundary[i, 1], boundary[i, 0])
        l = math.hypot(boundary[i, 0], boundary[i, 1]) - offset
        boundary_y[i] = math.sin(a) * l
        boundary_x[i] = math.cos(a) * l

    # mask all triangles with at least one point outside of the boundaries
    for icell in range(triangles.shape[0]):
        for iedge in range(3):
            if not point_in_polygon(boundary_x, boundary_y, x_coords_proj[triangles[icell, iedge]], y_coords_proj[triangles[icell, iedge]]):
                mask[icell] = True
                break


def __get_triangulation(projection, transformation, lon, lat, calculate_mask=True, cache=True):
    """
    calculate a triangulation for a given unstructured grid.

    Parameters
    ----------
    projection : cartopy.crs.Projection
            the target projection for the plot

    transformation : cartopy.crs.Projection
            the source projection of the coordinates.

    lon : xarray.DataArray or np.ndarray
            longitudes of the grid

    lat : xarray.DataArray or np.ndarray
            latitudes of the grid

    Returns
    -------
    mtri.Triangulation
            the calculation triangulation with points outside of the projection masked
    """
    # we need the coordinates as numpy arrays
    lon, lat = np.asarray(lon), np.asarray(lat)

    # is the result in cache?
    if cache:
        cache_key = (projection, transformation, sum(lon[:10]), sum(lat[:10]))
        if cache_key in __triangulation_cache:
            return __triangulation_cache[cache_key]

    # find the boundary of the projection as well as the grid coordinates in projection space
    boundary = np.asarray(projection.boundary)
    proj_coords = projection.transform_points(transformation, lon, lat)

    # calculate the triangulation
    tri = mtri.Triangulation(lon, lat)

    # mask triangles outside the projection
    if calculate_mask:
        mask = np.zeros(tri.triangles.shape[0], dtype=np.bool)
        __mask_triangles(tri.triangles, proj_coords, boundary, mask)
        tri.set_mask(mask)

    # cache the result
    if cache:
        __triangulation_cache[cache_key] = tri
    return tri


@dispatch((xarray.DataArray, np.ndarray), (xarray.DataArray, np.ndarray), (xarray.DataArray, np.ndarray))
def contour(variable, lon, lat, **kwargs):
    """
    Create a plot from an array variable that does not include its coordinates.

    Parameters
    ----------
    variable : xarray.DataArray or np.ndarray
            the data to plot.

    lon : xarray.DataArray or np.ndarray
            longitude coordinate.

    lat : xarray.DataArray or np.ndarray
            latitude coordinate.

    """
    # is it a global dataset?
    transformation = ccrs.PlateCarree()
    is_global = __is_global(lon, lat)
    projection = kwargs.get("projection", None)
    if projection is None:
        if is_global:
            projection = ccrs.Mollweide()
        else:
            projection = ccrs.PlateCarree()

    # create the plot
    if not "figure" in kwargs:
        fig = plt.figure()
    else:
        fig = kwargs["figure"]
    if not "axes" in kwargs:
        subplot_args = kwargs.get("subplot_args", (111,))
        subplot_kwargs = kwargs.get("subplot_kwargs", {})
        ax = fig.add_subplot(*subplot_args, projection=projection, **subplot_kwargs)
    else:
        ax = kwargs["axes"]

    # construct arguments for the matplotlib contour
    contour_args = {"transform": transformation,
                    "cmap": "CMRmap_r",
                    "levels": __get_nice_levels(variable),
                    "extend": "both"}
    # copy all arguments not specific to this function to the contour arguments
    for arg, value in six.iteritems(kwargs):
        if arg not in ["filled", "colorbar", "gridlines", "gridline_labes", "projection"]:
            contour_args[arg] = value

    # decide on the plot type based on variable and coordinate dimension
    if variable.ndim == 1 and lon.ndim == 1 and lat.ndim == 1:
        # calculate the triangulation
        tri = __get_triangulation(projection, transformation, lon, lat)
        # create the plot
        if kwargs.get("filled", True):
            contour = ax.tricontourf(tri, variable, **contour_args)
        else:
            contour = ax.tricontour(tri, variable, **contour_args)

    else:
        # create a contour plot
        if kwargs.get("filled", True):
            contour = ax.contourf(lon, lat, variable, **contour_args)
        else:
            contour = ax.contour(lon, lat, variable, **contour_args)

    # add coastlines
    resolution = kwargs.get("coastlines", True)
    if resolution is not False:
        if isinstance(resolution, str):
            ax.coastlines(resolution)
        else:
            ax.coastlines()

    # add gridlines
    if kwargs.get("gridlines", False) is True:
        ax.gridlines(color="black", linestyle="dotted", draw_labels=kwargs.get("gridline_labels", False))

    # add a colorbar
    if kwargs.get("colorbar", True):
        divider = make_axes_locatable(ax)
        width = AxesY(ax, aspect=0.07)
        pad = Fraction(1.0, width)
        cb_ax = divider.append_axes("right", size=width, pad=pad, axes_class=Axes, frameon=False)
        fig.colorbar(contour, cax=cb_ax)

    if is_global:
        ax.set_global()

    return fig, ax


@dispatch(xarray.DataArray)
def contour(variable, lon_name=None, lat_name=None, **kwargs):
    """
    Create a plot from an xarray variable that includes coordinates.

    Examples
    --------

    >>> fig, ax = enstools.plot.contour(data["TOT_PREC"][0, :, :], coastlines="50m")    # doctest: +SKIP

    .. figure:: images/example_plot_icon_01.png

        24h ICON forecast for precipitation read from a grib2 file. Have a look at the script
        examples/example_plot_icon_01.py for more details.

    >>> fig, ax1 = enstools.plot.contour(data["PMSL"][0, :] / 100.0, gridlines=True, subplot_args=(121,))   # doctest: +SKIP
    >>> fig, ax2 = enstools.plot.contour(data["TOT_PREC"][0, :], figure=fig, subplot_args=(122,))           # doctest: +SKIP

    .. figure:: images/example_plot_icon_02.png

        24h ICON forecast for mean sea level pressure (left) and precipitation (right). The data was read from
        grib2 files on the native ICON grid and plotted without interpolation onto a regular grid. Have a look at the
        script examples/example_plot_icon_02.py

    Parameters
    ----------
    variable : xarray.DataArray
            the data to plot.

    lon_name : string
            name of the longitude coordinate in the case of non standard names.

    lat_name : string
            name of the latitude coordinate in the case of non standard names.

    """
    # select projection and transformation based for global and non-global datasets
    # get the coordinates
    if lon_name is not None:
        lon = variable.coords[lon_name]
    else:
        lon = None
        for lon_name in __lon_names:
            if lon_name in variable.coords:
                lon = variable.coords[lon_name]
                break
        if lon is None:
            raise ValueError("not longitude coordinate found, checked names: %s" % ", ".join(__lon_names))
    if lat_name is not None:
        lat = variable.coords[lat_name]
    else:
        lat = None
        for lat_name in __lat_names:
            if lat_name in variable.coords:
                lat = variable.coords[lat_name]
                break
        if lat is None:
            raise ValueError("not longitude coordinate found, checked names: %s" % ", ".join(__lon_names))

    # check the units of the coordinates
    if isinstance(lon, xarray.DataArray):
        if "units" in lon.attrs and lon.attrs["units"] == "radian":
            lon = np.rad2deg(lon)
    if isinstance(lat, xarray.DataArray):
        if "units" in lat.attrs and lat.attrs["units"] == "radian":
            lat = np.rad2deg(lat)

    # check the dimensions, only the dimensions of lon and lat are allowed
    coord_dim_names = []
    for dim_name in lon.dims:
        if dim_name not in coord_dim_names:
            coord_dim_names.append(dim_name)
    for dim_name in lat.dims:
        if dim_name not in coord_dim_names:
            coord_dim_names.append(dim_name)
    for dim_name in variable.dims:
        if dim_name not in coord_dim_names:
            raise ValueError("the variable has more dimensions (%s) then the horizontal coordinates (%s). Please call this function with a slice of the variable!" % (", ".join(variable.dims), ", ".join(coord_dim_names)))

    # do we need a mesh grid?
    if lon.ndim == 1 and variable.ndim == 2:
        lon, lat = np.meshgrid(lon, lat)

    # create the plot with the coordinates found
    return contour(variable, lon, lat, **kwargs)


# add kwargs to all contour functions doc
__contour_kwargs_doc = """
Other optional keyword arguments:

**kwargs
        *figure*: matplotlib.figure.Figure
            If provided, this figure instance will be used (and returned), otherwise a new
            figure will be created.

        *axes*: matplotlib.axes.Axes
            If provided, this axes instance will be used (e.g., of overplotting), otherwise a
            new axes object will be created.

        *subplot_args*: tuple
            Arguments passed on to add_subplot. These arguments are used only if no axes is provided.

        *subplot_kwargs*: dict
            Keyword arguments passed on to add_subplot. These arguments are used only if no axes is provided.

        *filled*: [*True* | *False*]
            If True a filled contour is plotted, which is the default

        *colorbar*: [*True* | *False*]
            If True, a colorbar is created. Default=True.

        *gridlines*: [*True* | *False*]
            If True, coordinate grid lines are drawn. Default=False

        *gridline_labes*: [*True* | *False*]
            Whether or not to label the grid lines. The default is not to label them.

        *coastlines*: [*True* | *False*]
            If True, coordinate grid lines are drawn. Default=True

        *projection*: [*None* | *cartopy.crs.Projection*]
            If not None, the Projection object is used to create the plot

All other arguments are forwarded to the matplotlib contour or contourf function.

Returns
-------
tuple
        (Figure, Axes) of the new plot is returned. The returned values may be reused in subsequent calls to 
        plot functions.
"""
for __func in six.itervalues(contour.funcs):
    __func.__doc__ += __contour_kwargs_doc
