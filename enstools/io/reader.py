import logging
from collections import OrderedDict
import xarray
import dask
import six
import os
import numpy as np
from multipledispatch import dispatch
import glob
from enstools.misc import has_ensemble_dim, add_ensemble_dim, is_additional_coordinate_variable, first_element
from .file_type import get_file_type
try:
    from .eccodes import read_grib_file
except ImportError:
    pass


@dispatch(six.string_types)
def read(filename, constant=None, **kwargs):
    """
    Read one or more input files

    Parameters
    ----------
    filename : str
            name of one individual file of unix shell-like file name pattern for multiple files

    constant : str
            name of a file containing constant variables.

    **kwargs
            all arguments accepted by xarray.open_dataset() of xarray.open_mfdataset()

    Returns
    -------
    xarray.Dataset
            in-memory representation of the content of the input file(s)
    """
    # is the filename a pattern?
    files = __expand_file_pattern(filename)
    if len(files) > 1 or constant is not None:
        return read(files, constant=constant, **kwargs)
    else:
        return __open_dataset(filename)


@dispatch((list, tuple))
def read(filenames, constant=None, merge_same_size_dim=False, members_by_folder=False, **kwargs):
    """
    Read multiple input files

    Parameters
    ----------
    filename : list of str or tuple of str
            names of individual files or filename pattern

    merge_same_size_dim : bool
            dimensions of the same size but with different names are merged together. This is sometimes useful to
            merge datasets from different file formats (e.g., grib and netcdf). Default: False.

    members_by_folder : bool
            interpret files in one subdirectory as data from the same ensemble member. This is useful if the ensemble
            member number if not stored within the input file. Defaut: False.

    constant : str
            name of a file containing constant variables.

    **kwargs
            all arguments accepted by xarray.open_dataset() of xarray.open_mfdataset()

    Returns
    -------
    xarray.Dataset
            in-memory representation of the content of the input file(s)
    """
    # create at first a list of all input files
    datasets = []
    # loop over all input files and create delayed read objects
    expanded_filenames = []
    parent_folders = []
    for filename in filenames:
        # is the filename a pattern?
        files = __expand_file_pattern(filename)
        for one_file in files:
            one_file = os.path.abspath(one_file)
            datasets.append(dask.delayed(read)(one_file))
            expanded_filenames.append(one_file)
            parent = os.path.dirname(one_file)
            if not parent in parent_folders:
                parent_folders.append(parent)

    # read all the files in parallel
    datasets = dask.compute(*datasets, traverse=False, get=dask.multiprocessing.get)

    # are there ensemble members in different folders?
    if members_by_folder and len(parent_folders) > 1:
        # sort all parent folders and assign ensemble member numbers to them
        parent_folders = sorted(parent_folders)
        ens_member_by_folder = __assign_ensemble_member_number_to_folders(parent_folders)

        # create the ensemble dimension within the datasets
        for ids in range(len(datasets)):
            if not has_ensemble_dim(datasets[ids]):
                add_ensemble_dim(datasets[ids], ens_member_by_folder[os.path.dirname(expanded_filenames[ids])])

    # if dimensions have the same size but different names, then merge them by renaming
    if merge_same_size_dim:
        size_name_mapping = {}
        rename_mapping = {}
        for ds in datasets:
            for dim_name, dim_size in six.iteritems(ds.sizes):
                if dim_size not in size_name_mapping:
                    size_name_mapping[dim_size] = dim_name
                else:
                    if dim_name != size_name_mapping[dim_size]:
                        rename_mapping[dim_name] = size_name_mapping[dim_size]
        if len(rename_mapping) > 0:
            for ds in datasets:
                rename_mapping_for_ds = {}
                for old_name, new_name in six.iteritems(rename_mapping):
                    if old_name in ds.dims:
                        rename_mapping_for_ds[old_name] = new_name
                if len(rename_mapping_for_ds) > 0:
                    ds.rename(rename_mapping_for_ds, inplace=True)

    # combine datsets from different files
    result = __merge_datasets(datasets)

    # is there a file with constant data?
    if constant is not None:
        # read constant data
        constant_data = __open_dataset(constant)
        # remove time axes if present
        if "time" in constant_data.dims:
            constant_data = constant_data.isel(time=0)
        # remove variables also present in the non constant data
        for one_var in constant_data.data_vars:
            if one_var in result or one_var.lower() in result or one_var.upper() in result:
                logging.warning("variable '%s' in constant and non-constant data => renamed to '%s_constant'!", one_var, one_var)
                constant_data.rename({one_var: "%s_constant" % one_var}, inplace=True)
        # merge constant data
        result = xarray.auto_combine((result, constant_data))

    return result


def __merge_datasets(datasets):
    """

    Parameters
    ----------
    datasets

    Returns
    -------

    """
    # find common coordinates and sort variables according to those coordinates
    # get all coordinates
    all_coords = set()
    for ds in datasets:
        for coord in ds.coords.keys():
            if coord not in all_coords and np.issubdtype(ds.coords[coord].dtype, np.number):
                all_coords.add(coord)
    # filter for coordinates present in all files
    coords_in_all_files = OrderedDict()
    for coord in all_coords:
        in_all = True
        ascending = True
        for ds in datasets:
            if coord in ds.coords:
                # multi-dim coords are not sortable
                if ds.coords[coord].ndim > 1:
                    in_all = False
                    break
                if ds.coords[coord].size > 1:
                    if ds.coords[coord][0] > ds.coords[coord][1]:
                        ascending = False
            else:
                in_all = False
                break
        if in_all:
            coords_in_all_files[coord] = ascending

    # find coords with different values in different files
    for coord in six.iterkeys(coords_in_all_files):
        # loop over all files
        datasets_with_one_coord_value = {}
        for one_ds in datasets:
            first_value = first_element(one_ds.coords[coord])
            if first_value not in datasets_with_one_coord_value:
                datasets_with_one_coord_value[first_value] = [one_ds]
            else:
                datasets_with_one_coord_value[first_value].append(one_ds)
        # are there values with more than one dataset?
        different_values = sorted(datasets_with_one_coord_value.keys())
        if not coords_in_all_files[coord]:
            different_values = list(reversed(different_values))
        # merge files with more than one value
        if len(different_values) > 1:
            new_datasets = []
            for one_value in different_values:
                new_datasets.append(__merge_datasets(datasets_with_one_coord_value[one_value]))
            datasets = tuple(new_datasets)

    # sort the dataset by all shared coordinates
    for coord, ascending in six.iteritems(coords_in_all_files):
        datasets = sorted(datasets, key=lambda x: first_element(x.coords[coord]))

    # are there datasets with ensemble_members attribute?
    ensemble_members_found = []
    for one_dataset in datasets:
        if "ensemble_member" in one_dataset.attrs:
            if not one_dataset.attrs["ensemble_member"] in ensemble_members_found:
                ensemble_members_found.append(one_dataset.attrs["ensemble_member"])

    # data from more than one ensemble member was found. create the ensemble dimension for all datasets
    if len(ensemble_members_found) > 1:
        for one_dataset in datasets:
            if "ensemble_member" in one_dataset.attrs:
                add_ensemble_dim(one_dataset, one_dataset.attrs["ensemble_member"], inplace=True)
            else:
                logging.warning("Trying to merge ensemble and non-ensemble data. That will possibly fail!")
        return __merge_datasets(datasets)
    else:
        # try to merge the datasets
        return xarray.auto_combine(datasets)


def __open_dataset(filename):
    """
    read one input file. the type is automatically determined.

    Parameters
    ----------
    filename : str
            path of the file

    Returns
    -------
    xarray.Dataset
    """

    # get the type of the file
    file_type = get_file_type(filename)
    if file_type is None:
        raise ValueError("unable to guess the type of the input file '%s'" % filename)

    if file_type == "NC":
        result = xarray.open_dataset(filename, chunks={})
    elif file_type == "GRIB":
        result = read_grib_file(filename, debug=False)
    else:
        raise ValueError("unknown file type '%s' for file '%s'" % (file_type, filename))

    # check for additional coordinate variables like staggered lat/lon values
    for one_name, one_var in six.iteritems(result.data_vars):
        if is_additional_coordinate_variable(one_var):
            result.set_coords(one_name, True)
    return result


def __expand_file_pattern(pattern):
    """
    use glob to find all files matching the pattern

    Parameters
    ----------
    pattern : str
            unix bash shell like search pattern

    Returns
    -------
    list
            list of file names matching the pattern
    """
    if "*" in pattern or "?" in pattern or "[" in pattern or "{" in pattern:
        files = glob.glob(pattern)
        return files
    else:
        return [pattern]


def __assign_ensemble_member_number_to_folders(folders):
    """
    used internally to extract numbers from the folders and use them as ensemble member numbers. If not all folders
    contain numbers, or if they are not unique, then create consecutive numbers.

    Parameters
    ----------
    folders : list
            list of absolute paths

    Returns
    -------
    dict :
            dictionary where the key is the absolute path and the value if the ensemble member number
    """

    # extract all digits from the last parts of the folders
    digits = []
    for ifolder, folder in enumerate(folders):
        prefix, one_path = os.path.split(folder)
        digits_for_path = ""
        for letter in one_path:
            if letter.isdigit():
                digits_for_path += letter
        if digits_for_path == "":
            digits_for_path = ifolder + 1
        else:
            digits_for_path = int(digits_for_path)
        digits.append(digits_for_path)

    # are all numbers unique?
    digits = np.unique(digits)
    if len(digits) < len(folders):
        digits = np.linspace(1, len(folders), len(folders), dtype=np.int32)

    # create the result dictionary
    result = {}
    for ifolder in range(len(folders)):
        result[folders[ifolder]] = digits[ifolder]
    return result