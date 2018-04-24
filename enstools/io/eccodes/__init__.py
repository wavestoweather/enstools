"""
preliminary minimal support for grib files. This module will be replaced as soon as the official eccodes xarray-
support is available.
"""
import logging
try:
    from . import eccodes_cffi
except OSError:
    logging.warning("eccodes c-library not found, grib file support not available!")
    pass
from collections import OrderedDict
import os
import re
import xarray
import dask.array
import numpy
from datetime import datetime, timedelta
import logging


def read_grib_file(filename, debug=False, in_memory=False, leadtime_from_filename=False):
    """
    Read the contents of a grib1 or grib2 file

    Parameters
    ----------
    filename : string
            name of the file to read

    debug :  bool
            show some additional debug messages

    in_memory : bool
            load the data directly into memory

    leadtime_from_filename : bool
            COSMO-GRIB1-Files do not contain exact times. If this argument is set, then the timestamp is calculated
            from the init time and the lead time from the file name.

    Returns
    -------
    xarray.Dataset
            a netcdf-like representation of the file-content
    """
    if "eccodes_cffi" not in globals():
        raise ImportError("eccodes interface not found, grib file support not available!")

    # use always the absolute path, the relative path may change during the lifetime of the dataset
    filename = os.path.abspath(filename)

    # lists of dimensions coordinates and variables
    variables = set()
    levels = {}
    level_values = {}
    dimensions = {}
    dimension_names = {}
    coordinates = OrderedDict()
    attributes = {}
    encodings = {}
    ensemble_members = set()
    times = set()
    datatype = {}
    msg_by_var_level_ens = {}
    rotated_pole = {}

    # list of skipped grid types
    skipped_grids = set()

    # loop to select all messages
    logging.debug("start reading all grib messages from %s ..." % filename)
    gfile = open(filename, "rb")
    while True:
        # read the content of the next grib message from the input file.
        offset = gfile.tell()
        header_only_message = eccodes_cffi.read_message_header_bytes(gfile, offset, False)
        if header_only_message is None:
            break

        # create an empty message from the message raw data
        msg = eccodes_cffi.GribMessage(header_only_message)

        # skip messages on unsupported grids
        if msg["gridType"] not in ["sh", "regular_ll", "rotated_ll", "reduced_gg", "unstructured_grid"]:
            if msg["gridDefinitionDescription"] not in skipped_grids:
                skipped_grids.add(msg["gridDefinitionDescription"])
                logging.warning("skipping grib message due to unsupported grid: %s" % msg["gridDefinitionDescription"])
            continue

        # record the name
        variable_id = (msg.get_name(prefer_cf=False), msg["typeOfLevel"])
        if variable_id not in variables:
            variables.add(variable_id)

        # make a list of all levels
        if variable_id not in levels:
            levels[variable_id] = [msg["level"]]
            level_values[variable_id] = {}
        else:
            if msg["level"] not in levels[variable_id]:
                levels[variable_id].append(msg["level"])
        if msg["level"] not in level_values[variable_id]:
            level_values[variable_id][msg["level"]] = msg.get_level()

        # record the horizontal dimensions
        if variable_id not in dimensions:
            dimensions[variable_id], dimension_names[variable_id] = msg.get_dimension(dimensions, dimension_names)

        # record the valid time stamp
        if leadtime_from_filename:
            leadtime = re.search("lfff(\d\d)(\d\d)(\d\d)(\d\d)", os.path.basename(filename))
            if leadtime is None:
                raise IOError("unable to read leadtime from filename: %s", filename)
            initDate = "%08d%04d" % (msg["dataDate"], int(msg["dataTime"]))
            if initDate.startswith("0000"):
                initDate = "2" + initDate[1:]
            time_stamp = datetime.strptime(initDate, "%Y%m%d%H%M")
            time_stamp += timedelta(days=int(leadtime.group(1)), hours=int(leadtime.group(2)), minutes=int(leadtime.group(3)), seconds=int(leadtime.group(4)))
        else:
            validityDate = "%08d%04d" % (msg["validityDate"], int(msg["validityTime"]))
            if validityDate.startswith("0000"):
                validityDate = "2" + validityDate[1:]
            time_stamp = datetime.strptime(validityDate, "%Y%m%d%H%M")
        if time_stamp not in times:
            times.add(time_stamp)

        # horizontal coordinates
        if "lon" not in coordinates:
            coord_lon, coord_lat = msg.get_coordinates(dimension_names[variable_id])
            if coord_lon is not None and coord_lat is not None:
                coordinates["lon"], coordinates["lat"] = coord_lon, coord_lat

        # create a list of all ensemble members
        for ensemble_member_key in ["localActualNumberOfEnsembleNumber", "perturbationNumber"]:
            if ensemble_member_key in msg:
                ensemble_member = msg[ensemble_member_key]
                if msg[ensemble_member_key] not in ensemble_members:
                    ensemble_members.add(msg[ensemble_member_key])
                break
            else:
                ensemble_member = -1

        # select a datatype based on the number of bits per value in the grib message
        if variable_id not in datatype:
            if msg["bitsPerValue"] > 32:
                datatype[variable_id] = numpy.float64
            else:
                datatype[variable_id] = numpy.float32

        # collect attributes of this variables
        if variable_id not in attributes:
            attrs = OrderedDict()
            attrs["units"] = msg["parameterUnits"]
            attrs["long_name"] = msg["parameterName"]
            # add alternative names
            if "cfName" in msg and msg["cfName"] != "unknown":
                attrs["standard_name"] = msg["cfName"]
            if "cfVarName" in msg and msg["cfVarName"] != "unknown":
                attrs["cf_short_name"] = msg["cfVarName"]
            if "shortName" in msg and msg["shortName"] != "unknown":
                attrs["short_name"] = msg["shortName"]

            attrs["grid_type"] = msg["gridType"]
            # information about the rotated pole?
            if msg["gridType"] == "rotated_ll":
                rotated_pole[variable_id] = msg.get_rotated_ll_info(dimension_names[variable_id])
                attrs["grid_mapping"] = rotated_pole[variable_id][0]
            elif msg["gridType"] in ["reduced_gg", "unstructured_grid"]:
                attrs["coordinates"] = "lon lat"
            attributes[variable_id] = attrs
            # values used for encoding during storage
            encoding = OrderedDict()
            encoding["_FillValue"] = datatype[variable_id](msg["missingValue"])
            encodings[variable_id] = encoding

        # store the values in a dict for later retrieval
        if not in_memory:
            msg_by_var_level_ens[(variable_id, msg["level"], ensemble_member, time_stamp)] = \
                dask.array.from_delayed(dask.delayed(__get_one_message)(filename, offset, dimensions[variable_id], datatype[variable_id], encodings[variable_id]["_FillValue"]),
                                        shape=dimensions[variable_id],
                                        dtype=datatype[variable_id])
        else:
            # persist the data into memory
            msg_by_var_level_ens[(variable_id, msg["level"], ensemble_member, time_stamp)] = \
                                        dask.array.from_array(msg.get_values(dimensions[variable_id], datatype[variable_id], encodings[variable_id]["_FillValue"]),
                                        chunks=dimensions[variable_id])

    logging.debug("finish reading all grib messages from %s, start construction of arrays..." % filename)

    # create the coordinate definition for the dataset
    if len(ensemble_members) > 0:
        coordinates["ens"] = ("ens", numpy.array(sorted(list(ensemble_members)), dtype=numpy.int32))
    coordinates["time"] = ("time", sorted(list(times)))

    # add vertical coordinates
    level_bounds = {}
    level_coordinates = {}
    level_coordinate_names = {}
    for one_var in variables:
        values = level_values[one_var]
        level_coordinate = numpy.zeros((len(values),), dtype=numpy.float32)
        # create bounds of coordinate or only coordinate
        level_bound = None
        if type(values[levels[one_var][0]]) == tuple:
            level_bound = numpy.zeros((len(values), 2), dtype=numpy.float32)
            for ilevel, one_level in enumerate(sorted(levels[one_var])):
                level_bound[ilevel, :] = values[one_level]
                level_coordinate[ilevel] = level_bound[ilevel, :].mean()
        else:
            for ilevel, one_level in enumerate(sorted(levels[one_var])):
                level_coordinate[ilevel] = values[one_level]
        # is the name of the level already in use?
        coordinate_name = one_var[1]
        coordinate_name_counter = 0
        while coordinate_name in level_coordinates:
            if level_coordinates[coordinate_name].size == level_coordinate.size and numpy.all(level_coordinates[coordinate_name] == level_coordinate):
                break
            coordinate_name_counter += 1
            coordinate_name = "%s%d" % (one_var[1], coordinate_name_counter)
        if coordinate_name not in level_coordinates:
            level_coordinates[coordinate_name] = level_coordinate
        level_coordinate_names[one_var] = coordinate_name
        if level_bound is not None:
            level_bounds["%s_bnds" % coordinate_name] = level_bound

    # create the actual data arrays in a loop over all variables
    xarray_variables = OrderedDict()
    for one_var in variables:
        if debug:
            print(one_var)
        # calculate the dimensions for this variable
        var_levels = sorted(levels[one_var])
        var_dims = dimensions[one_var]
        var_shape_ = [len(coordinates["time"][1]), len(ensemble_members), len(var_levels)]
        var_shape_.extend(var_dims)
        var_dim_names_ = ["time", "ens"]
        if one_var in level_coordinate_names:
            var_level_coordinate_name = level_coordinate_names[one_var]
        else:
            var_level_coordinate_name = one_var[1]
        var_dim_names_.append(var_level_coordinate_name)
        var_dim_names_.extend(dimension_names[one_var])
        var_shape = []
        var_dim_names = []
        var_has_meaningful_levels = var_level_coordinate_name in level_coordinates \
                                    and (level_coordinates[var_level_coordinate_name][0] != 0 or
                                         level_coordinates[var_level_coordinate_name].size > 1) \
                                    and ("%dm" % level_coordinates[var_level_coordinate_name][0] not in one_var[0].lower())
        for ishape in range(len(var_shape_)):
            if var_shape_[ishape] > 1 or var_dim_names_[ishape] == "time" or \
                    (var_dim_names_[ishape] in level_coordinates and var_has_meaningful_levels):
                var_shape.append(var_shape_[ishape])
                var_dim_names.append(var_dim_names_[ishape])

        # loop over all timesteps
        stacked_times = []
        for one_time in coordinates["time"][1]:
            # loop over all ensemble members
            stacked_ensemble = []
            if "ens" in coordinates:
                ensemble_members = coordinates["ens"][1]
            else:
                ensemble_members = [-1]
            for one_ens in ensemble_members:
                # loop over all levels
                stacked_levels = []
                for one_level in var_levels:
                    msg_key = (one_var, one_level, one_ens, one_time)
                    if msg_key in msg_by_var_level_ens:
                        stacked_levels.append(msg_by_var_level_ens[(one_var, one_level, one_ens, one_time)])
                    else:
                        stacked_levels.append(dask.array.from_delayed(
                            dask.delayed(numpy.full)(var_dims, encodings[variable_id]["_FillValue"]),
                            shape=dimensions[variable_id],
                            dtype=datatype[variable_id]))

                # combine all levels into one array
                if len(stacked_levels) > 1:
                    stacked_levels = dask.array.stack(stacked_levels)
                elif var_has_meaningful_levels:
                    stacked_levels = dask.array.reshape(stacked_levels[0], (1,) + stacked_levels[0].shape)
                else:
                    stacked_levels = stacked_levels[0]

                # create a list of all ensemble members
                stacked_ensemble.append(stacked_levels)

            # combine all ensemble members into one array
            if len(stacked_ensemble) > 1:
                stacked_ensemble = dask.array.stack(stacked_ensemble)
            else:
                stacked_ensemble = stacked_ensemble[0]

            # create a list of all times
            stacked_times.append(stacked_ensemble)

        # combine all times into one array
        if len(stacked_times) > 1:
            da_var = dask.array.stack(stacked_times)
        else:
            da_var = dask.array.reshape(stacked_times[0], (1,) + stacked_times[0].shape)

        # finally create the xarray and add it to the dataset
        # use the level only in the name if the name is otherwise not unique
        name_is_unique = True
        for one_var_compare in variables:
            if one_var_compare[0] == one_var[0] and one_var_compare[1] != one_var[1]:
                name_is_unique = False
                break
        if name_is_unique:
            var_name = "%s" % one_var[0]
        else:
            var_name = "%s_%s" % (one_var)

        # time, ens, and horizontal coordinates
        var_coords = OrderedDict()
        for coord in coordinates:
            if coord in var_dim_names or "n%s" % coord in var_dim_names or "r%s" % coord in var_dim_names:
                var_coords[coord] = coordinates[coord]

        # vertical coordinate
        if var_has_meaningful_levels:
            var_coords[var_level_coordinate_name] = level_coordinates[var_level_coordinate_name]

        # create the xarray object
        xarray_variables[var_name] = xarray.DataArray(da_var,
                                                      dims=var_dim_names,
                                                      name=var_name,
                                                      attrs=attributes[one_var],
                                                      coords=var_coords,
                                                      encoding=encodings[one_var])

        # are there lon-lat values?
        if "cell" in var_dim_names[-1] and "lon" not in xarray_variables and "lon" in coordinates:
            xarray_variables["lon"] = coordinates["lon"]
            xarray_variables["lat"] = coordinates["lat"]

        # is there a rotated pole for this variable?
        if one_var in rotated_pole and rotated_pole[one_var][0] not in xarray_variables:
            xarray_variables[rotated_pole[one_var][0]] = rotated_pole[one_var][1]
            xarray_variables[var_dim_names[-2]] = rotated_pole[one_var][2]
            xarray_variables[var_dim_names[-1]] = rotated_pole[one_var][3]

    # add all bounds arrays to the dataset
    for one_bounds in level_bounds.keys():
        xarray_variables[one_bounds] = xarray.DataArray(level_bounds[one_bounds],
                                                        dims=(one_bounds.replace("_bnds", ""), "bnds"))

    # sort the variables by name
    xarray_variables_sorted = OrderedDict(sorted(xarray_variables.items(), key=lambda i: i[0].lower()))

    # create the dataset object
    logging.debug("creating xarray dataset for %s ..." % filename)
    dataset = xarray.Dataset(xarray_variables_sorted)
    if "lon" in dataset:
        dataset["lon"].attrs["long_name"] = "longitude"
        dataset["lon"].attrs["units"] = "degrees_east"
    if "lat" in dataset:
        dataset["lat"].attrs["long_name"] = "latitude"
        dataset["lat"].attrs["units"] = "degrees_north"

    # data from one ensemble member? store as attribute
    if "ens" in coordinates and len(coordinates["ens"][1]) == 1:
        dataset.attrs["ensemble_member"] = coordinates["ens"][1][0]

    # add bounds attribute to coordinates
    for one_bounds in level_bounds.keys():
        coordinate_name = one_bounds.replace("_bnds", "")
        if coordinate_name in dataset:
            dataset[coordinate_name].attrs["bounds"] = one_bounds
    logging.debug("finished reading %s" % filename)

    return dataset


def __get_one_message(filename, offset, shape, dtype, missing):
    """
    get the values of the message at position *imsg* from a grib file

    Parameters
    ----------
    filename
    imsg

    Returns
    -------
    numpy.ndarray
    """
    # open the input file, seek the message and read it
    with open(filename, "rb") as gfile:
        # read raw message
        content = eccodes_cffi.read_message_header_bytes(gfile, offset, read_data=True)

        # decode the message
        msg = eccodes_cffi.GribMessage(content)

        # decode the actual values
        return msg.get_values(shape, dtype, missing)

