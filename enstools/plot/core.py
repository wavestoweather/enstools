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

from enstools.core import check_arguments
from enstools.misc import point_in_polygon
import xarray
import six
import math
import cartopy
import warnings


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
    if lon.max() - lon.min() > 350 and lat.max() - lat.min() > 170:
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


def get_nice_levels(variable, center_on_zero=False, nlevel=11, min_percentile=0.1, max_percentile=99.9):
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
    minval, maxval = np.nanpercentile(variable, [min_percentile, max_percentile])
    perc10 = __get_order_of_magnitude(maxval - minval) / 10.0

    # adjust minimum, especially around zero
    minval_is_adjusted = False
    if minval > 0 > minval - perc10:
        minval = 0
        minval_is_adjusted = True
    elif minval < 0 < maxval and abs(minval) < abs(maxval) / 1000:
        minval = 0
        minval_is_adjusted = True
    elif center_on_zero:
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
    if center_on_zero:
        minval = -maxval
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


def get_coordinates_from_xarray(variable, lon_name=None, lat_name=None, create_mesh=True, only_spatial_dims=True, rad2deg=True):
    """
    get coordinate arrays from a xarray object

    Parameters
    ----------
    variable : xarray.DataArray
            xarray with coordinates

    lon_name : str
            name of the longitude coordinate

    lat_name : str
            name of the latitude coordinate

    rad2deg: bool
            convert radian to degrees. True by default.

    Returns
    -------
    tuple:
            lon, lat arrays
    """
    # check arguments
    if isinstance(lon_name, (np.ndarray, xarray.DataArray)) and isinstance(lat_name, (np.ndarray, xarray.DataArray)):
        return lon_name, lat_name
    if not isinstance(variable, xarray.DataArray):
        raise ValueError("named coordinates are only allowed for xarray variable!")

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
    if rad2deg:
        if isinstance(lon, xarray.DataArray):
            if "units" in lon.attrs and lon.attrs["units"] == "radian":
                lon = np.rad2deg(lon)
        if isinstance(lat, xarray.DataArray):
            if "units" in lat.attrs and lat.attrs["units"] == "radian":
                lat = np.rad2deg(lat)

    # check the dimensions, only the dimensions of lon and lat are allowed
    if only_spatial_dims:
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
    if create_mesh and lon.ndim == 1 and variable.ndim == 2 and lon_name != "rlon" and lat_name != "rlat":
        lon, lat = np.meshgrid(lon, lat)
    return lon, lat


@check_arguments(dims={'variable': ('lat', 'lon')})
def contour(variable, lon=None, lat=None, **kwargs):
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

    lon : xarray.DataArray or np.ndarray or str
            longitude coordinate or name of longitude coordinate. The name may only be used for xarray variables.

    lat : xarray.DataArray or np.ndarray or str
            latitude coordinate or name of longitude coordinate. The name may only be used for xarray variables.

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

            *colorbar*: [*True* | *False* | "empty"]
                If True, a colorbar is created. Default=True. Use empty to reserve space for the colorbar
                without actually creating it. This is usefull for multipanel plots, where one panel has a colorbar and
                another not.

            *levels*: np.ndarray
                If provided, these levels are used, otherwise the levels are automatically selected.

            *levels_center_on_zero* : bool
                If true, automatically selected levels are centered around zero.

            *gridlines*: [*True* | *False*]
                If True, coordinate grid lines are drawn. Default=False

            *gridline_labes*: [*True* | *False*]
                Whether or not to label the grid lines. The default is not to label them.

            *coastlines*: [*True* | *False* | '110m' | '50m' | '10m']
                If True, coordinate grid lines are drawn. Default=True

            *coastlines_kwargs*: dict
                dictionary with arguments passed on to ax.coastlines()

            *borders*: [*True* | *False* | '110m' | '50m' | '10m']
                If True, coordinate grid lines are drawn. Default=False

            *projection*: [ *cartopy.crs.Projection*]
                If not None, the Projection object is used to create the plot

            *rotated_pole*: [*xarray.DataArray* | dict]
                Information about the rotated pole. This can either be the CF-standard rotated_pole variable from
                an input file, or alternatively a dictionary with the keys grid_north_pole_latitude and
                grid_north_pole_longitude.

    All other arguments are forwarded to the matplotlib contour or contourf function.

    Returns
    -------
    tuple
            (Figure, Axes) of the new plot is returned. The returned values may be reused in subsequent calls to
            plot functions.
    """
    # get the coordinates
    lon, lat = get_coordinates_from_xarray(variable, lon, lat)

    # is it a global dataset?
    transformation = ccrs.PlateCarree()
    is_global = __is_global(lon, lat)
    projection = kwargs.get("projection", None)
    if projection is None:
        if is_global:
            projection = ccrs.Mollweide()
        else:
            projection = ccrs.PlateCarree()

    # is it a rotated pole array
    if kwargs.get("rotated_pole", None) is not None:
        rotated_pole = kwargs.get("rotated_pole")
        if isinstance(rotated_pole, xarray.DataArray):
            grid_north_pole_latitude = rotated_pole.grid_north_pole_latitude
            grid_north_pole_longitude = rotated_pole.grid_north_pole_longitude
        elif isinstance(rotated_pole, dict):
            grid_north_pole_latitude = rotated_pole["grid_north_pole_latitude"]
            grid_north_pole_longitude = rotated_pole["grid_north_pole_longitude"]
        else:
            raise ValueError("unsupported type of rotated_pole argument!")

        # create a projection if no projection was specified
        if not "projection" in kwargs:
            projection = ccrs.RotatedPole(grid_north_pole_longitude, grid_north_pole_latitude)

        # create a transformation if the coordinates are 1d
        if lon.ndim == 1:
            transformation = projection

    # create the plot
    if kwargs.get("figure", None) is None:
        fig = plt.figure()
    else:
        fig = kwargs.pop("figure")
    if kwargs.get("axes", None) is None:
        subplot_args = kwargs.pop("subplot_args", (111,))
        subplot_kwargs = kwargs.pop("subplot_kwargs", {})
        ax = fig.add_subplot(*subplot_args, projection=projection, **subplot_kwargs)
    else:
        ax = kwargs.pop("axes")

    # construct arguments for the matplotlib contour
    contour_args = {"transform": transformation,
                    "extend": "both"}
    if kwargs.get("levels", None) is None:
        contour_args["levels"] = get_nice_levels(variable, center_on_zero=kwargs.get("levels_center_on_zero", False))
    else:
        contour_args["levels"] = kwargs["levels"]
    if kwargs.get("cmap", None) is not None:
        contour_args["cmap"] = kwargs["cmap"]
    elif kwargs.get("colors", None) is None:
        contour_args["cmap"] = "CMRmap_r"

    # copy all arguments not specific to this function to the contour arguments
    for arg, value in six.iteritems(kwargs):
        if arg not in ["coastlines", "coastlines_kwargs", "borders", "borders_kwargs", "filled", "colorbar", "gridlines", "gridline_labes", "projection", "levels_center_on_zero", "rotated_pole", "cmap"]:
            contour_args[arg] = value

    # decide on the plot type based on variable and coordinate dimension
    # FIXME: suppress a specific warning about MaskedArrays
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', lineno=6385)  # suppress a specific warning about MaskedArrays
        warnings.filterwarnings('ignore', lineno=6442)  # suppress a specific warning about MaskedArrays

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
            # transpose if necessary, lat is expected to be the first dimension of the variable
            # TODO: make that more reliable! Possibly by reordering of the input variable.
            if np.all(lon.shape == np.flip(variable.shape, 0)) and not lon.shape[0] == lon.shape[1]:
                variable = variable.transpose()
            if kwargs.get("filled", True):
                contour = ax.contourf(lon, lat, variable, **contour_args)
            else:
                contour = ax.contour(lon, lat, variable, **contour_args)

    # add coastlines
    # TODO: ensure that coastlines or borders are not added multiple times
    resolution_coastlines = kwargs.get("coastlines", True)
    if resolution_coastlines is not False:
        coastlines_kwargs = kwargs.get("coastlines_kwargs", dict())
        if isinstance(resolution_coastlines, str):
            ax.coastlines(resolution_coastlines, **coastlines_kwargs)
        else:
            ax.coastlines(**coastlines_kwargs)

    # add borders
    resolution_borders = kwargs.get("borders", False)
    if not isinstance(resolution_borders, str) and isinstance(resolution_coastlines, str):
        resolution_borders = resolution_coastlines
    if resolution_borders is not False:
        if isinstance(resolution_borders, str):
            ax.add_feature(cartopy.feature.NaturalEarthFeature(
                "cultural", "admin_0_boundary_lines_land",
                resolution_borders, edgecolor='black', facecolor='none', linestyle=":"))
        else:
            ax.add_feature(cartopy.feature.BORDERS, linestyle=":")

    # add gridlines
    if kwargs.get("gridlines", False) is True:
        ax.gridlines(color="black", linestyle="dotted", draw_labels=kwargs.get("gridline_labels", False))

    # add a colorbar
    if kwargs.get("colorbar", True) is True or kwargs.get("colorbar", True) == "empty":
        divider = make_axes_locatable(ax)
        width = AxesY(ax, aspect=0.07)
        pad = Fraction(1.0, width)
        cb_ax = divider.append_axes("right", size=width, pad=pad, axes_class=Axes, frameon=False)
        if kwargs.get("colorbar", True) is True:
            fig.colorbar(contour, cax=cb_ax)
        else:
            cb_ax.tick_params(bottom="off", left="off", top="off", right="off", which="both", labelbottom="off", labelleft="off")

    if is_global:
        ax.set_global()

    return fig, ax
