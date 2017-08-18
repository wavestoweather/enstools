"""
preliminary minimal support for grib files. This module will be replaced as soon as the official eccodes xarray-
support is available.
"""
from collections import OrderedDict
import os
import xarray
import dask.array
import numpy
import eccodes
from datetime import datetime
import logging

# cache for open grib messages
__cache = {}


class HeaderOnlyGribFile(eccodes.GribFile):

    def next(self, headers_only=True):
        try:
            return self.MessageClass(self, headers_only=headers_only)
        except IOError:
            raise StopIteration()


def read_grib_file(filename, debug=False):
    """
    Read the contents of a grib1 or grib2 file

    Parameters
    ----------
    filename : string
            name of the file to read

    Returns
    -------
    xarray.Dataset
            a netcdf-like representation of the file-content
    """
    # open the file
    grib = HeaderOnlyGribFile(filename)

    # lists of dimensions coordinates and variables
    variables = set()
    levels = {}
    dimensions = {}
    dimension_names = {}
    coordinates = OrderedDict()
    attributes = {}
    ensemble_members = set()
    times = set()
    isConstant = {}
    datatype = {}
    msg_by_var_level_ens = {}

    # list of skipped grid types
    skipped_grids = set()

    # find the variables and array dimensions in a loop over all grib messages
    #__cache[os.path.abspath(filename)] = []
    for imsg, msg in enumerate(grib):
        # cache the message for later usage
        #__cache[os.path.abspath(filename)].append(msg)
        # print debug information
        if imsg == 0 and debug:
            for one_key in sorted(msg.keys()):
                if one_key.startswith("md5"):
                    continue
                if isinstance(msg[one_key], numpy.ndarray):
                    print("%40s %s %s" % (one_key, type(msg[one_key]), msg[one_key].shape))
                else:
                    print("%40s %s" % (one_key, msg[one_key]))

        # skip messages on unsupported grids
        if msg["gridType"] not in ["sh", "rotated_ll", "reduced_gg", "unstructured_grid"]:
            if msg["gridDefinitionDescription"] not in skipped_grids:
                skipped_grids.add(msg["gridDefinitionDescription"])
                logging.warning("skipping grib message due to unsupported grid: %s" % msg["gridDefinitionDescription"])
            continue

        # record the name
        variable_id = (msg["cfVarName"], msg["typeOfLevel"])
        if not variable_id in variables:
            variables.add(variable_id)

        # is this variable constant?
        if variable_id not in isConstant:
            if msg["isConstant"] != 0:
                isConstant[variable_id] = True
            else:
                isConstant[variable_id] = False

        # make a list of all levels
        if variable_id not in levels:
            levels[variable_id] = [msg["level"]]
        else:
            if not msg["level"] in levels[variable_id]:
                levels[variable_id].append(msg["level"])

        # record the horizontal dimensions
        if variable_id not in dimensions:
            dimensions[variable_id], dimension_names[variable_id] = __get_dimension_for_message(msg, dimensions, dimension_names)

        # record the valid time stamp
        if isConstant[variable_id]:
            time_stamp == "constant"
        else:
            time_stamp = datetime.strptime("%s%04d" % (msg["validityDate"], int(msg["validityTime"])), "%Y%m%d%H%M")
            if time_stamp not in times:
                times.add(time_stamp)

        # horizontal coordinates
        if "lon" not in coordinates:
            coordinates["lon"], coordinates["lat"] = __get_coordinates_for_message(msg, dimension_names[variable_id])

        # create a list of all ensemble members
        for ensemble_member_key in ["localActualNumberOfEnsembleNumber", "perturbationNumber"]:
            if ensemble_member_key in msg:
                ensemble_member = msg[ensemble_member_key]
                if msg[ensemble_member_key] not in ensemble_members:
                    ensemble_members.add(msg[ensemble_member_key])
                break
            else:
                ensemble_member = -1

        # collect attributes of this variables
        if variable_id not in attributes:
            attrs = OrderedDict()
            attrs["units"] = msg["parameterUnits"]
            attrs["long_name"] = msg["parameterName"]
            attrs["_FillValue"] = msg["missingValue"]
            attrs["short_name"] = msg["shortName"]
            attributes[variable_id] = attrs

        # select a datatype based on the number of bits per value in the grib message
        if not variable_id in datatype:
            if msg["bitsPerValue"] > 32:
                datatype[variable_id] = numpy.float64
            else:
                datatype[variable_id] = numpy.float32

        # store the values in a dict for later retrieval
        msg_by_var_level_ens[(variable_id, msg["level"], ensemble_member, time_stamp)] = \
            dask.array.from_delayed(dask.delayed(__get_one_message)(filename, imsg),
                                    shape=dimensions[variable_id],
                                    dtype=datatype[variable_id])
        # close the message
        msg.close()

    # create the coordinate definition for the dataset
    if len(ensemble_members) > 0:
        coordinates["ens"] = ("ens", sorted(list(ensemble_members)))
    coordinates["time"] = ("time", sorted(list(times)))

    # create the actual data arrays in a loop over all variables
    xarray_variables = {}
    for one_var in variables:
        if debug:
            print(one_var)
        # calculate the dimensions for this variable
        var_levels = sorted(levels[one_var])
        var_dims = dimensions[one_var]
        var_shape_ = [len(coordinates["time"][1]), len(ensemble_members), len(var_levels)]
        var_shape_.extend(var_dims)
        var_dim_names_ = ["time", "ens", "levels"]
        var_dim_names_.extend(dimension_names[one_var])
        var_shape = []
        var_dim_names = []
        for ishape in range(len(var_shape_)):
            if var_shape_[ishape] > 1 or (var_dim_names_[ishape] == "time" and not isConstant[one_var]):
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
                            dask.delayed(numpy.full)(var_dims, attributes[variable_id]["_FillValue"]),
                            shape=dimensions[variable_id],
                            dtype=datatype[variable_id]))

                # combine all levels into one array
                if len(stacked_levels) > 1:
                    stacked_levels = dask.array.stack(stacked_levels)
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
        if not isConstant[one_var]:
            if len(stacked_times) > 1:
                da_var = dask.array.stack(stacked_times)
            else:
                da_var = dask.array.reshape(stacked_times[0], (1,) + stacked_times[0].shape)
        else:
            da_var = stacked_times[0]

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
        var_coords = OrderedDict()
        for coord in coordinates:
            if coord in var_dim_names:
                var_coords[coord] = coordinates[coord]
        xarray_variables[var_name] = xarray.DataArray(da_var,
                                                      dims=var_dim_names,
                                                      name=var_name,
                                                      attrs=attributes[one_var],
                                                      coords=var_coords)

    dataset = xarray.Dataset(xarray_variables)
    if "lon" in dataset:
        dataset["lon"].attrs["long_name"] = "longitude"
        dataset["lon"].attrs["units"] = "degrees_east"
    if "lat" in dataset:
        dataset["lat"].attrs["long_name"] = "latitude"
        dataset["lat"].attrs["units"] = "degrees_north"
    return dataset


def __get_one_message(filename, imsg):
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
    if os.path.abspath(filename) in __cache:
        msg = __cache[os.path.abspath(filename)][imsg]
        shape, _ = __get_dimension_for_message(msg)
        return msg["values"].reshape(shape)
    else:
        grib = eccodes.GribFile(filename)
        __cache[os.path.abspath(filename)] = []
        for i, msg in enumerate(grib):
            __cache[os.path.abspath(filename)].append(msg)
            if i == imsg:
                shape, _ = __get_dimension_for_message(msg)
                values = msg["values"].reshape(shape)
        return values


def __get_dimension_for_message(msg, dimensions=None, dimension_names=None):
    """
    get the shape of one message depending on the grid type

    Parameters
    ----------
    msg : GribMessage

    Returns
    -------
    tuple
            (shape, dim-names)
    """
    if msg["gridType"] in ["rotated_ll"]:
        shape = (msg["Ni"], msg["Nj"])
        dim_names = ["nlon", "nlat"]
    elif msg["gridType"] in ["sh", "reduced_gg", "unstructured_grid"]:
        shape = (msg["numberOfValues"],)
        dim_names = ["ncells"]
    else:
        raise ValueError("don't know how to calculate the shape for grid type %s" % msg["gridType"])

    # loop over all already used dims for comparison
    if dimensions is not None and dimension_names is not None:
        for one_var in dimensions.keys():
            if dimension_names[one_var] == dim_names and dimensions[one_var] != shape:
                for id, dn in enumerate(dim_names):
                    dim_names[id] = "%s%d" % (dn, 2)
    return shape, dim_names


def __get_coordinates_for_message(msg, dimension_names):
    """
    get the longitude and latitude coordinates for one message

    Parameters
    ----------
    msg : GribMessage

    Returns
    -------
    tuple:
        ((lon-dim-names, lon-coord), (lat-dim-names), lat-coord)
    """
    # are coordinates available?
    if "longitudes" not in msg or "latitudes" not in msg:
        return None, None

    if msg["gridType"] == "rotated_ll":
        lon = (dimension_names, msg["longitudes"].reshape(msg["Ni"], msg["Nj"]))
        lat = (dimension_names, msg["latitudes"].reshape(msg["Ni"], msg["Nj"]))
    elif msg["gridType"] == "reduced_gg":
        lon = (dimension_names, msg["longitudes"])
        lat = (dimension_names, msg["latitudes"])
    else:
        lon = (dimension_names[0], msg["longitudes"])
        lat = (dimension_names[1], msg["latitudes"])
    return lon, lat
