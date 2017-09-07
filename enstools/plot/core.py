"""
Basic plot routines usable to construct more advanced plots
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from numba import jit
from multipledispatch import dispatch
import xarray


# names for coordinates
from shapely.geometry.multipoint import MultiPoint

__lon_names = ["lon", "lons", "longitude", "longitudes", "clon", "rlon"]
__lat_names = ["lat", "lats", "latitude", "latitudes", "clat", "rlat"]


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


def __get_nice_levels(variable, nlevel=11, min_percentile=0.5, max_percentile=99.5):
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
def __mask_triangles(lon, lat, triangles, proj_coords, mask):
    x_coords_proj = proj_coords[:, 0]
    y_coords_proj = proj_coords[:, 1]

    min_x = x_coords_proj.min()
    max_x = x_coords_proj.max()
    man_length = 0.1

    min_y = y_coords_proj.min() / 10.0
    max_y = y_coords_proj.max() / 10.0

    mean_length = 0.0
    for icell in range(triangles.shape[0]):
        l1 = np.sqrt((x_coords_proj[triangles[icell, 0]] - x_coords_proj[triangles[icell, 1]])**2 + (y_coords_proj[triangles[icell, 0]] - y_coords_proj[triangles[icell, 1]])**2)
        l2 = np.sqrt((x_coords_proj[triangles[icell, 0]] - x_coords_proj[triangles[icell, 2]])**2 + (y_coords_proj[triangles[icell, 0]] - y_coords_proj[triangles[icell, 2]])**2)
        l3 = np.sqrt((x_coords_proj[triangles[icell, 1]] - x_coords_proj[triangles[icell, 2]])**2 + (y_coords_proj[triangles[icell, 1]] - y_coords_proj[triangles[icell, 2]])**2)
        mean_length += l1 + l2 + l3
    mean_length = mean_length / (triangles.shape[0] * 3.0)

    for icell in range(triangles.shape[0]):
        l1 = np.sqrt((x_coords_proj[triangles[icell, 0]] - x_coords_proj[triangles[icell, 1]])**2 + (y_coords_proj[triangles[icell, 0]] - y_coords_proj[triangles[icell, 1]])**2)
        l2 = np.sqrt((x_coords_proj[triangles[icell, 0]] - x_coords_proj[triangles[icell, 2]])**2 + (y_coords_proj[triangles[icell, 0]] - y_coords_proj[triangles[icell, 2]])**2)
        l3 = np.sqrt((x_coords_proj[triangles[icell, 1]] - x_coords_proj[triangles[icell, 2]])**2 + (y_coords_proj[triangles[icell, 1]] - y_coords_proj[triangles[icell, 2]])**2)
        if l1 < mean_length / 10.0 or l2 < mean_length / 10.0 or l3 < mean_length / 10.0:
            mask[icell] = 1
        if l1 > mean_length * 10.0 or l2 > mean_length * 10.0 or l3 > mean_length * 10.0:
            mask[icell] = 1

                #in_east = False
        #in_west = False
        #in_north = False
        #in_south = False
        #for iedge in range(3):
        #    if x_coords_proj[triangles[icell, iedge]] > max_x:
        #        in_east = True
        #    if x_coords_proj[triangles[icell, iedge]] < min_x:
        #        in_west = True
        #    if y_coords_proj[triangles[icell, iedge]] > max_y:
        #        in_north = True
        #    if y_coords_proj[triangles[icell, iedge]] < min_y:
        #        in_south = True
        #mask[icell] = (in_east and in_west) or (in_north and in_south)

                    #if lon[triangles[icell, iedge]] > 179.0 or lon[triangles[icell, iedge]] < -179.0:
            #    mask[icell] = True
            #    break
            #if lat[triangles[icell, iedge]] > 87.0 or lat[triangles[icell, iedge]] < -87.0:
            #    mask[icell] = True
            #    break

@dispatch((xarray.DataArray, np.ndarray), (xarray.DataArray, np.ndarray), (xarray.DataArray, np.ndarray))
def map_plot(variable, lon, lat, **kwargs):
    """
    Create a plot from an array variable that does not include its coordinates.

    Parameters
    ----------
    variable : xarray.DataArray or np.ndarray

    lon : xarray.DataArray or np.ndarray
            longitude coordinate

    lat : xarray.DataArray or np.ndarray
            latitude coordinate

    Optional keyword arguments:

    **kwargs
            *colorbar*: [*True* | *False]
                If True, a colorbar is created. Default=True.

            *gridlines*: [*True* | *False*]
                If True, coordinate grid lines are drawn. Default=False

            *gridline_labes*: [*True* | *False*]
                Whether or not to label the grid lines. The default is not to label them.

            *projection*: [*None* | *cartopy.crs.Projection *]
                If not None, the Projection object is used to create the plot

    Returns
    -------
    GeoAxes
            the axes object of the new plot is returned.
    """
    # is it a global dataset?
    transformation = ccrs.PlateCarree()
    is_global = __is_global(lon, lat)
    projection = kwargs.get("projection", None)
    if projection is None:
        if is_global:
            projection = ccrs.Robinson()
        else:
            projection = ccrs.PlateCarree()

    # create the plot
    ax = plt.axes(projection=projection)

    # decide on the plot type based on variable and coordinate dimension
    if variable.ndim == 1 and lon.ndim == 1 and lat.ndim == 1:
        # calculate the triangulation
        #lon = np.where(lon > 179.5, 179.5, lon)
        #lon = np.where(lon < -179.5, -179.5, lon)

        proj_coords = projection.transform_points(transformation, np.asarray(lon), np.asarray(lat))
        tri = mtri.Triangulation(lon, lat)
        mask = np.zeros(tri.triangles.shape[0], dtype=np.bool)

        #print(proj_coords.shape)

        __mask_triangles(np.asarray(lon), np.asarray(lat), tri.triangles, proj_coords, mask)
        tri.set_mask(mask)
        #print(np.any(mask))

        contour = ax.tricontourf(tri, variable,
                                 transform=transformation,
                                 cmap="CMRmap_r",
                                 levels=__get_nice_levels(variable),
                                 extend="both")
    else:
        # create a contour plot
        contour = ax.contourf(lon, lat, variable,
                              transform=transformation,
                              cmap="CMRmap_r",
                              levels=__get_nice_levels(variable),
                              extend="both")
    # add coastlines
    ax.coastlines()

    # add gridlines
    if kwargs.get("gridlines", False) is True:
        ax.gridlines(color="black", linestyle="dotted", draw_labels=kwargs.get("gridline_labels", False))

    # add a colorbar
    if kwargs.get("colorbar", True):
        plt.colorbar(contour, ax=ax, shrink=.6)

    if is_global:
        ax.set_global()

    return ax


@dispatch(xarray.DataArray)
def map_plot(variable, lon_name=None, lat_name=None, **kwargs):
    """
    Create a plot from an xarray variable that includes coordinates.

    Parameters
    ----------
    variable : xarray.DataArray

    lon_name : string
            name of the longitude coordinate in the case of non standard names

    lat_name : string
            name of the latitude coordinate in the case of non standard names

    Optional keyword arguments:

    **kwargs
            *colorbar*: [*True* | *False]
                If True, a colorbar is created. Default=True.

            *gridlines*: [*True* | *False*]
                If True, coordinate grid lines are drawn. Default=False

            *gridline_labes*: [*True* | *False*]
                Whether or not to label the grid lines. The default is not to label them.

            *projection*: [*None* | *cartopy.crs.Projection *]
                If not None, the Projection object is used to create the plot

    Returns
    -------
    GeoAxes
            the axes object of the new plot is returned.
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
    return map_plot(variable, lon, lat, **kwargs)
