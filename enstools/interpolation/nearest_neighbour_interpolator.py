from enstools.core import check_arguments
import numpy as np
import xarray
import scipy.spatial
from collections import OrderedDict


class NearestNeighbourInterpolator:
    """
    This class performs the actual interpolation. It is initialised by nearest_neighbour, not directly
    """
    def __init__(self, indices, weights, distances, shape, n_target_points, n_source_points, target_shape, target_lon, target_lat):
        self._indices = indices
        self._weights = weights
        self._distances = distances
        self._shape = shape
        self._n_target_points = n_target_points
        self._n_source_points = n_source_points
        self._target_shape = target_shape
        self._target_lon = target_lon
        self._target_lat = target_lat

    def __call__(self, data):
        """
        Interpolate the data given on the input grid to the points specified during the creation of the interpolator.

        Parameters
        ----------
        data : np.ndarray of xarray.DataArray
                Input array with the same shape as defined by the coordinates used to create the interpolator

        Returns
        -------
        np.ndarray or xarray.DataArray with the shape of the point coordinates used in the creation of the interpolator.
        If the array to interpolate has more dimensions, then these dimensions are prepended to the result, e.g.
        (level, points).
        """
        # check the shape of the data array
        if data.shape[-len(self._shape):] != self._shape:
            raise ValueError("the rightmost dimension of the array to interpolate from must be %s" % (self._shape,))

        # how man dimensions should the result have?
        if len(data.shape) == len(self._shape):
            result = np.empty(self._n_target_points)
        else:
            result = np.empty(data.shape[:-len(self._shape)] + (self._n_target_points,))

        if len(self._shape) == 1:
            if self._n_source_points == 1:
                result[..., :] = data[..., self._indices]
            else:
                for i in range(self._n_target_points):
                    data_values = data[..., self._indices[i]]
                    result[..., i] = np.sum(data_values * self._weights[i, ...], axis=-1)
        elif len(self._shape) == 2:
            if self._n_source_points == 1:
                if self._n_target_points == 1:
                    result[..., 0] = data[..., self._indices[0], self._indices[1]]
                else:
                    for i in range(self._n_target_points):
                        result[..., i] = data[..., self._indices[0][i], self._indices[1][i]]
            else:
                for i in range(self._n_target_points):
                    data_values = data[..., self._indices[0][i], self._indices[1][i]]
                    result[..., i] = np.sum(data_values * self._weights[i, ...], axis=-1)

        # reshape if necessary
        if data.ndim > len(self._shape):
            target_shape = data.shape[:data.ndim-len(self._shape)] + self._target_shape
        else:
            target_shape = self._target_shape
        if result.shape != target_shape:
            result = result.reshape(target_shape)

        # create xarray dataset
        result_coords = OrderedDict()
        result_attrs = OrderedDict()
        result_name = "interpolated"
        if isinstance(data, xarray.DataArray):
            result_attrs = data.attrs
            result_dims = data.dims[:data.ndim-len(self._shape)]
            for one_dim in result_dims:
                result_coords[one_dim] = data.coords[one_dim]
            result_name = data.name
        else:
            if len(result.shape) > 1:
                result_dims = ("dim_0",)
                for d in range(1, data.ndim-len(self._shape)):
                    result_dims += ("dim_%d" % d,)
            else:
                result_dims = ()
        if len(self._target_shape) == 2:
            result_dims += ("lat", "lon")
            result_coords["lat"] = self._target_lat
            result_coords["lon"] = self._target_lon
            result_attrs["grid_type"] = "regular_ll"
        else:
            result_dims += ("cell",)
            result_coords["lat"] = xarray.DataArray(np.asarray(self._target_lat), dims=("cell",))
            result_coords["lon"] = xarray.DataArray(np.asarray(self._target_lon), dims=("cell",))
            result_attrs["grid_type"] = "unstructured_grid"
            result_attrs["coordinates"] = "lon lat"
        result = xarray.DataArray(result, dims=result_dims, coords=result_coords, attrs=result_attrs, name=result_name)
        return result


@check_arguments(units={"src_lon": "degrees_east",
                        "src_lat": "degrees_north",
                        "dst_lon": "degrees_east",
                        "dst_lat": "degrees_north"})
def nearest_neighbour(src_lon, src_lat, dst_lon, dst_lat, src_grid="regular", dst_grid="unstructured", npoints=1, method="mean"):
    """
    Find the coordinates of station locations within gridded model data. Supported are 1d- and 2d-coordinates of regular
    grids (e.g. rotated lat-lon) or 'unstructured' grids like the ICON grid.

    Parameters
    ----------
    src_lon : np.ndarray or xarray.DataArray
            1d or 2d coordinate in x-direction of the source grid

    src_lat : np.ndarray or xarray.DataArray
            1d or 2d coordinate in y-direction of the source grid

    dst_lon : np.ndarray or xarray.DataArray
            1d coordinate in x-direction of the station locations

    dst_lat : np.ndarray or xarray.DataArray
            1d coordinate in y-direction of the station locations

    src_grid : {'regular', 'unstructured'}
            Type of input grid. Possible values are:
            "regular": a regular grid with 1d or 2d coordinates. 1d-coordinates are internally converted to 2d-
            coordinates using meshgrid. This selection is the default.
            "unstructured": The grid is given as a 1d list of points (e.g., station data or ICON model output).

    dst_grid : {'regular', 'unstructured'}
            Type of output grid. Possible values are:
            "regular": a regular grid with 1d coordinates.
            "unstructured": The grid is given as a 1d list of points (e.g., station data or ICON model output). This
            selection is the default.

    npoints : int
            Number of nearest points to be used in the interpolation. For a regular grid useful values are 4 or 12. For
            data on the ICON grid 3 or 6 might be used. The actual geometry of the grid is not taken into account. The
            points are selected based merely on distance.

    method : {'mean', 'd-mean'}
            "mean" : the mean value of the neighbour points is used with equal weight.
            "d-mean": each point is weighted by the reciprocal of the squared distance. The minimum distance within this
            calculation is half of the mean grid spacing.

    Returns
    -------
    NearestNeighbourInterpolator
            callable interpolator object. Each call returns interpolated values

    Examples
    --------
    >>> import numpy
    >>> lon = numpy.arange(10)
    >>> lat = numpy.arange(15)
    >>> gridded_data = numpy.zeros((15, 10))
    >>> gridded_data[8, 4] = 3
    >>> f = nearest_neighbour(lon, lat, 4.4, 7.6)
    >>> f(gridded_data)
    <xarray.DataArray 'interpolated' (cell: 1)>
    array([ 3.])
    Coordinates:
        lat      (cell) float64 7.6
        lon      (cell) float64 4.4
    Dimensions without coordinates: cell
    Attributes:
        grid_type:    unstructured_grid
        coordinates:  lon lat
    """
    # create an array containing all coordinates
    if src_grid == "regular":
        if src_lon.ndim == 1:
            lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        elif src_lon.ndim == 2:
            if src_lon.shape != src_lat.shape:
                raise ValueError("for 2d-coordinates, the shapes have to match!")
            lon_2d, lat_2d = src_lon, src_lat
        else:
            raise ValueError("only 1d- and 2d-coordinates are supported for regular grids")
        coords = np.stack((lon_2d.flatten(), lat_2d.flatten()), axis=1)
        input_dims = lon_2d.shape
    elif src_grid == "unstructured":
        if src_lon.ndim == 1:
            coords = np.stack((src_lon, src_lat), axis=1)
            input_dims = src_lon.shape
        else:
            raise ValueError("an unstructured grid is supposed to have 1d-coordinate arrays!")

    # convert point coordinates if given as scalar
    if not hasattr(dst_lon, "__len__"):
        dst_lon = np.array((dst_lon,))
        dst_lat = np.array((dst_lat,))

    # keep the target coordinates
    target_lon = dst_lon
    target_lat = dst_lat

    # create coordinates for regular output grid
    if dst_grid == "regular" and dst_lon.ndim == 1 and dst_lat.ndim == 1:
        point_lon_2d, point_lat_2d = np.meshgrid(dst_lon, dst_lat)
        dst_lon = point_lon_2d.flatten()
        dst_lat = point_lat_2d.flatten()
        target_shape = point_lon_2d.shape
    else:
        target_shape = (len(dst_lon),)

    # create the kd-tree and calculate the indices and the weights for the indices
    kdtree = scipy.spatial.cKDTree(coords)

    # estimate the distance between grid points by calculating the distance to the next neighbour for three points
    grid_point_distance, _ = kdtree.query(coords[[0, len(coords) // 2, len(coords)-1]], k=2)
    mean_grid_point_distance = grid_point_distance[:, 1].mean()

    # calculate indices and distances for the target points
    if len(dst_lon) == 1:
        distances, indices_flat = kdtree.query((dst_lon[0], dst_lat[0]), k=npoints)
    else:
        distances, indices_flat = kdtree.query(np.stack((dst_lon, dst_lat), axis=1), k=npoints)
    if npoints > 1 and len(distances.shape) == 1:
        distances = np.expand_dims(distances, 0)
        indices_flat = np.expand_dims(indices_flat, 0)
    if npoints > 1:
        if method == "mean":
            weights = np.empty(distances.shape)
            weights[:] = 1.0 / npoints
        elif method == "d-mean":
            weights = 1.0 / np.maximum(distances, mean_grid_point_distance / 2.0)**2
            weights = weights / np.tile(np.expand_dims(np.sum(weights, 1), 1), npoints)
        else:
            raise ValueError("unsupported method: %s" % method)
    else:
        weights = 1
    if len(input_dims) > 1:
        indices = np.unravel_index(indices_flat, input_dims)
    else:
        indices = indices_flat
    if type(indices) == int:
        indices = np.asarray([indices])

    # construct and return the interpolator object
    return NearestNeighbourInterpolator(indices, weights, distances, input_dims, len(dst_lon), npoints, target_shape, target_lon, target_lat)
