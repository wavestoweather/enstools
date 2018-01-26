#
# this code calculates vertical averages of a row of patches
#
from enstools import io
import xarray as xr
import numpy as np
import sys
from collections import OrderedDict
import pdb
#
##
PY2 = sys.version_info[0] < 3
PY3 = sys.version_info[0] >= 3
if PY3:  # pragma: no cover
    def iteritems(d):
        return iter(d.items())

    def itervalues(d):
        return iter(d.values())

    range = range
else:  # pragma: no cover
    # Python 2
    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()

    range = xrange

# ======================================================================
# ======================================================================


def swap_dimsN(self, dims_dict, inplace=False):
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


def staggered_grid_interpolation(self, label='upper'):
    if isinstance(self, xr.core.dataset.Dataset):
        variables = OrderedDict()
        for name, var in iteritems(self.variables):
            if "srlon" in var.dims:
                # prepare slices
                kwargs_start = {"srlon": slice(None, -1)}
                kwargs_end = {"srlon": slice(1, None)}
                # prepare new coordinate
                if label == 'upper':
                    kwargs_new = kwargs_end
                elif label == 'lower':
                    kwargs_new = kwargs_start
                else:
                    raise ValueError('The \'label\' argument has to be either '
                                     '\'upper\' or \'lower\'')
                if name in self.data_vars:
                    variables[name] = ((var.isel(**kwargs_end) + var.isel(**kwargs_start)) / 2.)
                else:
                    variables[name] = var.isel(**kwargs_new)
            elif "srlat" in var.dims:
                # prepare slices
                kwargs_start = {"srlat": slice(None, -1)}
                kwargs_end = {"srlat": slice(1, None)}
                # prepare new coordinate
                if label == 'upper':
                    kwargs_new = kwargs_end
                elif label == 'lower':
                    kwargs_new = kwargs_start
                else:
                    raise ValueError('The \'label\' argument has to be either '
                                     '\'upper\' or \'lower\'')
                if name in self.data_vars:
                    variables[name] = ((var.isel(**kwargs_end) + var.isel(**kwargs_start)) / 2.)
                else:
                    variables[name] = var.isel(**kwargs_new)
            else:
                variables[name] = var
        result = self._replace_vars_and_dims(variables)
        return result
    else:
        raise TypeError('FileType for staggered_grid_interpolation has to be either a xarray Dataset')


def interp_velocity(self, label='upper'):
    if isinstance(self, xr.core.dataset.Dataset):
        for name, var in iteritems(self.data_vars):
            if "srlon" in var.dims:
                ds = staggered_grid_interpolation(var._to_temp_dataset(), label=label)
                self[name] = var._from_temp_dataset(ds)
            elif "srlat" in var.dims:
                ds = staggered_grid_interpolation(var._to_temp_dataset(), label=label)
                self[name] = var._from_temp_dataset(ds)
            else:
                pass
        self = swap_dimsN(self, {"srlon": "rlon"})
        self = swap_dimsN(self, {"srlat": "rlat"})
        return self
    elif isinstance(self, xr.core.dataarray.DataArray):
        ds = staggered_grid_interpolation(self._to_temp_dataset(), label=label)
        return self._from_temp_dataset(ds)
    else:
        raise TypeError('FileType has to be either a xarray Dataset or DataArray')


if __name__ == '__main__':

    ds = io.read('/project/meteo/w2w/B3/FBaur_output/2009063000/output_cosmoDE_2MOM_chess_alternate_glob_025_20_mgrid/lfff00120000z.nc')
    pdb.set_trace()
    ds = interp_velocity(ds)
    pdb.set_trace()




