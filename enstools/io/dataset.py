# some domain specific service routines for xarray.Datasets
import six


def drop_unused(ds):
    """
    COSMO is one example of a model which stores all possible coordinates within the output files even if no data
    variables are present which would use these variables. This function checks all coordinates and removes those that
    are not used by any data variable. Unused dimensions are also removed from the dataset.

    Some special coordinates are kept:
    - rotated_pole

    Parameters
    ----------
    ds : xarray.Dataset
            Dataset to check.

    Returns
    -------
    xarray.Dataset:
            a copy of the input dataset with removed coordinates.
    """
    # list of unremovable coordinates
    unremovable = ["rotated_pole"]

    # loop over all coordinates
    coords_to_remove = []
    for one_coord_name, one_coord_var in six.iteritems(ds.coords):
        # skip unremoveable coordinates
        if one_coord_name in unremovable:
            continue

        # this coordinate is only kept if all its dimensions are used in any of the data variables
        has_all_dims_in_any_var = False
        for one_name, one_var in six.iteritems(ds.data_vars):
            has_all_dims = len(one_coord_var.dims) > 0
            for one_dim in one_coord_var.dims:
                if not one_dim in one_var.dims:
                    has_all_dims = False
                    break
            if has_all_dims:
                has_all_dims_in_any_var = True
                break

        # mark this coord for deletion?
        if not has_all_dims_in_any_var:
            coords_to_remove.append(one_coord_name)

    # actually remove the coords
    if len(coords_to_remove) > 0:
        for one_coord_name in coords_to_remove:
            if one_coord_name in ds.indexes:
                ds = ds.reset_index(one_coord_name, drop=True)
            elif one_coord_name in ds.coords:
                ds = ds.reset_coords(one_coord_name, drop=True)

    # remove unused dimensions
    unused_dims = []
    for one_dim in ds.dims:
        is_used = False
        for one_name, one_var in six.iteritems(ds.data_vars):
            if one_dim in one_var.dims:
                is_used = True
                break
        if not is_used:
            unused_dims.append(one_dim)

    # actually remove unused dims. This is not straight forward: it is necessary to create a dummy variable. That will
    # trigger recalculation of the dimensions.
    if len(unused_dims) > 0:
        ds["dummy_for_triggering_dims_calculation"] = 0
        del ds["dummy_for_triggering_dims_calculation"]

    return ds