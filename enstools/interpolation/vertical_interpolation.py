from numba import jit
import numpy as np
import xarray
from enstools.misc import swapaxis
from collections import OrderedDict
import logging


class model2pressure:

    def __init__(self, src_p, dst_p, vertical_dim=None):
        """
        Create an interpolator object for the interpolation from model to pressure level

        Parameters
        ----------
        src_p : xarray.DataArray or np.ndarray
                full model pressure at all grid cells. 2d for unstructured grid and 3d for regular grid. The rightmost
                dimension is expected to be the level coordinate, e.g. (lev, lat, lon) or (lev, cell)

        dst_p : xarray.DataArray or np.ndarray or float
                pressure level(s) to interpolate to

        vertical_dim : int
                the position of the vertical axes in the input data. if not specified, it is automatically detected. In
                this case, for numpy arrays, it has to be the first axes.

        Returns
        -------
        model2pressure
                callable interpolator object.
        """

        # find the vertical dimension of the input array
        if vertical_dim is None:
            if isinstance(src_p, xarray.DataArray):
                vertical_dim_names = ["pres", "p", "lev", "level", "isobaric", "layer", "hybrid"]
                for dimi, dim in enumerate(src_p.dims):
                    for valid_name in vertical_dim_names:
                        if valid_name in dim.lower():
                            vertical_dim = dimi
                            break
                    if vertical_dim is not None:
                        break
                if vertical_dim is not None:
                    logging.info("using dimension %d (%s) as vertical dimension." % (vertical_dim, src_p.dims[vertical_dim]))
                else:
                    raise ValueError("unable to auto-detect vertical dimension in input data.")
            else:
                if src_p.ndim > 3:
                    raise ValueError("unsupported number of dimensions for numpy array: %d. Try (lev, lat, lon) or (lev, cell) or xarray." % src_p.ndim)
                else:
                    vertical_dim = 0
        self._src_vertical_dim = vertical_dim

        # reorder the array if the vertical dimension is not the first dimension
        self._src_shape_not_reordered = src_p.shape
        if self._src_vertical_dim > 0:
            src_p = swapaxis(src_p, self._src_vertical_dim, 0)

        # is the destination a single value?
        if not hasattr(dst_p, "__len__"):
            dst_p = np.asarray([dst_p], dtype=np.float64)

        # store the original shape of the source field and the expected destination field
        self._src_shape = src_p.shape
        self._dst_shape = (len(dst_p),) + src_p.shape[1:]

        # store attributes and coordinates of the source array
        if isinstance(src_p, xarray.DataArray):
            self._src_attrs = src_p.attrs
            self._src_dims = src_p.dims
            self._src_coords = src_p.coords
        else:
            self._src_attrs = OrderedDict()
            self._src_dims = ("lev",)
            for dim in range(1, src_p.ndim):
                self._src_dims += ("dim_%d" % dim,)
            self._src_coords = OrderedDict()

        # store the destination pressure as coordiinate for the result arrays
        self._dst_pressure = dst_p

        # if the input pressure field is 3d, then flatten to two leftmost dimensions
        if src_p.ndim > 2:
            src_p = np.asarray(src_p).reshape((src_p.shape[0], np.product(src_p.shape[1:])))

        # calculate the weights for each horizontal grid point
        self._indices, self._weights = get_weights(np.asarray(src_p), np.asarray(dst_p))

    def __call__(self, data):
        """
        perform the actual interpolation

        Parameters
        ----------
        data : xarray.DataArray or np.ndarray
                the data array to interpolate. The shape has to match the shape of the array used to create this object

        Returns
        -------
        xarray.DataArray
                the interpolated array (lev, lat, lon) or (lev, cell)
        """
        # check the shape of the input array
        if data.shape != self._src_shape_not_reordered:
            raise ValueError("the shape of the data array to interpolate has to match the shape of the original pressure array: %s" % str(self._src_shape_not_reordered))

        # are there attributes at the data field?
        if isinstance(data, xarray.DataArray):
            data_attrs = data.attrs
            data_name = data.name
        else:
            data_attrs = OrderedDict()
            data_name = "interpolated"

        # reorder the dimensions if required
        if self._src_vertical_dim > 0:
            data = swapaxis(data, self._src_vertical_dim, 0)

        # if the input data field is 3d, then flatten to two leftmost dimensions
        if data.ndim > 2:
            data = np.asarray(data).reshape((data.shape[0], np.product(data.shape[1:])))

        # perform the interpolation
        result_data = apply_weights(np.asarray(data), self._indices, self._weights)

        # reshape required?
        if result_data.shape != self._dst_shape:
            result_data = result_data.reshape(self._dst_shape)

        # create a xarray object
        result = xarray.DataArray(result_data, dims=("pressure",) + self._src_dims[1:], attrs=data_attrs, name=data_name)
        result.coords["pressure"] = self._dst_pressure
        # copy coordinates for the rightmost dimensions
        for dim in self._src_dims[1:]:
            if dim in self._src_coords:
                result.coords[dim] = self._src_coords[dim]

        # reorder required?
        if self._src_vertical_dim > 0:
            result = swapaxis(result, self._src_vertical_dim, 0)
        return result


@jit(nopython=True)
def apply_weights(data, indices, weights):
    """
    use the indices and weights to create a new interpolated array

    Parameters
    ----------
    data : np.ndarray
            the source data array

    indices : np.ndarray
            the indices array created by get_weights

    weights : np.ndarray
            the weights array created by get_weights

    Returns
    -------
    np.ndarray
            the interpolated array
    """
    # create the result array
    result = np.empty(indices.shape[0:2])

    # loop over all target levels
    for target in range(indices.shape[0]):
        # loop over all cells
        for cell in range(indices.shape[1]):
            # if no interpolation is possible, write a nan value
            result[target, cell] = 0
            valid = False
            for val in range(2):
                if weights[target, cell, val] > 0.0:
                    result[target, cell] += data[indices[target, cell, val], cell] * weights[target, cell, val]
                    valid = True
            if not valid:
                result[target, cell] = np.nan
    return result


@jit(nopython=True)
def get_weights(src_p, dst_p):
    """
    calculate indices and weights for each column

    Parameters
    ----------
    src_p : np.ndarray
            array with source pressure grid (lev, cell)

    dst_p : np.ndarray
            array with destination pressure grid (lev)

    Returns
    -------
    indices, weights : tuple
            tuple of arrays with shape (target_levels, cells, 2)
    """
    # arrays for the indices of the neighbouring grid points and the respective weights
    indices = np.empty((dst_p.shape[0], src_p.shape[1], 2), dtype=np.int64)
    weights = np.empty((dst_p.shape[0], src_p.shape[1], 2), dtype=np.float64)

    # loop over all target values
    for target in range(dst_p.shape[0]):
        # loop over all horizontal points
        for cell in range(src_p.shape[1]):
            # find the largest value smaller or equal than the target value and the
            # smallest values larger than the target value
            largest_smaller = -1
            smallerst_larger = -1
            for lev in range(src_p.shape[0]):
                if src_p[lev, cell] <= dst_p[target]:
                    if largest_smaller == -1 or src_p[largest_smaller, cell] < src_p[lev, cell]:
                        largest_smaller = lev
                if src_p[lev, cell] > dst_p[target]:
                    if smallerst_larger == -1 or src_p[smallerst_larger, cell] > src_p[lev, cell]:
                        smallerst_larger = lev

            # store the indices
            indices[target, cell, 0] = largest_smaller
            indices[target, cell, 1] = smallerst_larger

            # use the indices of the neighbour to calculate the weights
            if smallerst_larger != -1 and src_p[smallerst_larger, cell] == dst_p[target]:
                weights[target, cell, 0] = 1.0
                weights[target, cell, 1] = 0.0
            elif largest_smaller == -1 or smallerst_larger == -1:
                weights[target, cell, :] = 0.0
            else:
                weights[target, cell, 0] = 1.0 - (dst_p[target] - src_p[largest_smaller, cell]) / (src_p[smallerst_larger, cell] - src_p[largest_smaller, cell])
                weights[target, cell, 1] = 1.0 - (src_p[smallerst_larger, cell]  - dst_p[target]) / (src_p[smallerst_larger, cell] - src_p[largest_smaller, cell])

    return indices, weights
