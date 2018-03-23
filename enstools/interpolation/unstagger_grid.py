import xarray as xr
from collections import OrderedDict
from six import iteritems


def __swap_dims_n(self, dims_dict, inplace=False):
    for k, v in dims_dict.items():
        if k not in self.dims:
            raise ValueError('cannot swap from dimension %r because it is '
                             'not an existing dimension' % k)

    result_dims = set(dims_dict.get(dim, dim) for dim in self.dims)

    variables = OrderedDict()

    coord_names = self._coord_names.copy()
    coord_names.update(dims_dict.values())

    for k, v in iteritems(self.variables):
        dims = tuple(dims_dict.get(dim, dim) for dim in v.dims)
        if k in result_dims:
            var = v.to_index_variable()
        else:
            var = v.to_base_variable()
        var.dims = dims
        variables[k] = var

    return self._replace_vars_and_dims(variables, coord_names, inplace=inplace)


def __staggered_grid_interpolation(data, src_coord, dst_coord, label='upper'):
    """
    perform the actual interpolation on a Dataset
    
    Parameters
    ----------
    data : xarray.Dataset
            dataset with staggered variables

    src_coord : str
            name of staggered coordinate to unstagger

    dst_coord : str
            name of unstaggered destination variable

    label : {"upper", "lower"}
            upper: first value along the unstaggered dimension will be missing, all upper values will be present.
            lower: last value along the unstaggered dimension will be missing, all lower values will be present.

    Returns
    -------
    xarray.Dataset :
            unstaggered Dataset.
    """
    if isinstance(data, xr.core.dataset.Dataset):
        variables = OrderedDict()
        # prepare slices
        kwargs_start = {src_coord: slice(None, -1)}
        kwargs_end = {src_coord: slice(1, None)}
        # prepare new coordinate
        if label == 'upper':
            kwargs_new = kwargs_end  # {dst_coord: slice(1, None)}
        elif label == 'lower':
            kwargs_new = kwargs_start  # {dst_coord: slice(None, -1)}
        else:
            raise ValueError('The \'label\' argument has to be either '
                             '\'upper\' or \'lower\'')

        if dst_coord not in data.variables:
            variables[dst_coord] = ((data.variables[src_coord].isel(**kwargs_end) + data.variables[src_coord].isel(**kwargs_start)) / 2.)
            variables[dst_coord].name = dst_coord
            variables[dst_coord].dims = (dst_coord,)
        for name, var in iteritems(data.variables):
            if src_coord in var.dims:
                if name in data.data_vars:
                    variables[name] = ((var.isel(**kwargs_end) + var.isel(**kwargs_start)) / 2.)
                else:
                    variables[name] = var.isel(**kwargs_new)
            else:
                variables[name] = var
        result = data._replace_vars_and_dims(variables)
        return result
    else:
        raise TypeError('FileType for __staggered_grid_interpolation has to be either a xarray Dataset')


def unstagger(data):
    """
    Interpolate variables on staggered grid (e.g., U and V) to grid cell centers. Every model has different names for
    staggered coordinates and variables. Currently, only the names of the model COSMO are known and supported.

    Parameters
    ----------
    data : xarray.DataArray or xarray.DataSet
            Array or Dataset with staggered variables. For datasets, all staggered variables are unstaggered. The
            staggered coordinate arrays are removed

    Returns
    -------
    xarray.DataArray or xarray.DataSet
            The return type depends on the input type.
    """
    # check variable and select variable names
    staggered_coords = {}
    if "srlon" in data.coords:
        staggered_coords["srlon"] = ("rlon", "upper", "slonu", "slatu")
    if "srlat" in data.coords:
        staggered_coords["srlat"] = ("rlat", "upper", "slonv", "slatv")
    if len(staggered_coords) == 0:
        raise ValueError("no supported staggered coordinates found in input of function unstagger!")

    # loop over all staggered coordinates.
    for staggered_coord, unstaggered_coord in iteritems(staggered_coords):
        if isinstance(data, xr.core.dataset.Dataset):
            for name, var in iteritems(data.data_vars):
                if staggered_coord in var.dims:
                    ds = var._to_temp_dataset()
                    ds[unstaggered_coord[0]] = data[unstaggered_coord[0]]
                    ds = __staggered_grid_interpolation(ds, staggered_coord, unstaggered_coord[0], label=unstaggered_coord[1])
                    data[name] = var._from_temp_dataset(ds)
            data = __swap_dims_n(data, {staggered_coord: unstaggered_coord[0]})
        elif isinstance(data, xr.core.dataarray.DataArray):
            ds = __staggered_grid_interpolation(data._to_temp_dataset(), staggered_coord, unstaggered_coord[0], label=unstaggered_coord[1])
            ds = __swap_dims_n(ds, {staggered_coord: unstaggered_coord[0]})
            data = data._from_temp_dataset(ds)
        else:
            raise TypeError('Input type for unstagger has to be either a xarray.Dataset or xarray.DataArray')

    # remove staggered coordinates
    for staggered_coord, unstaggered_coord in iteritems(staggered_coords):
        for to_remove in unstaggered_coord[2:]:
            if to_remove in data.coords:
                del data[to_remove]
        if staggered_coord in data.coords:
            del data[staggered_coord]

    return data
