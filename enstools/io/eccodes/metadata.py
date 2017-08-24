try:
    import eccodes
except ImportError:
    pass
import numpy
import xarray
from .index import get_lock


class GribMessageMetadata:
    """
    Dictionary like interface to the metadata of an grib message
    """

    def __init__(self, iid):
        """
        Initialize the message object from an GribIndex

        Parameters
        ----------
        iid : int
                numeric id of the grib index containing this message
        """
        self.gid = eccodes.codes_new_from_index(iid)
        self.cache = {}

    def isvalid(self):
        """
        Returns
        -------
        bool
                False if the message was not initialized or is already released
        """
        return self.gid is not None

    def __getitem__(self, item):
        if item in self.cache:
            return self.cache[item]
        else:
            try:
                if eccodes.codes_get_size(self.gid, item) > 1:
                    value = eccodes.codes_get_array(self.gid, item)
                else:
                    value = eccodes.codes_get(self.gid, item)
                self.cache[item] = value
                return value
            except eccodes.KeyValueNotFoundError:
                pass
            # nothing found?
            raise KeyError("key '%s' not found in grib message!" % item)

    def __contains__(self, item):
        if item in self.cache:
            return True
        else:
            try:
                value = self.__getitem__(item)
                self.cache[item] = value
            except KeyError:
                return False
            return True

    def print_all_keys(self):
        """
        for debugging: print a list all keys along with their values
        """
        with get_lock():
            # create the iterator
            iterid = eccodes.codes_keys_iterator_new(self.gid)

            # loop over all keys
            print("-" * 100)
            while eccodes.codes_keys_iterator_next(iterid):
                key_name = eccodes.codes_keys_iterator_get_name(iterid)
                if key_name.startswith("md5"):
                    continue
                key_val = self.__getitem__(key_name)
                if isinstance(key_val, numpy.ndarray):
                    print("%40s = array%s" % (key_name, key_val.shape))
                else:
                    print("%40s = %s" % (key_name, key_val))

            # clean memory
            eccodes.codes_keys_iterator_delete(iterid)

    def get_name(self, prefer_cf=True):
        """
        find a name for this variable.

        Parameters
        ----------
        prefer_cf : bool
                if True, the search order for the name is "cfName", "cfVarName", "shortName", otherwise it is
                "shortName", "cfName", "cfVarName".

        Returns
        -------
        string
                name of the variable.
        """
        if prefer_cf:
            name_keys = ["cfName", "cfVarName", "shortName"]
        else:
            name_keys = ["shortName", "cfName", "cfVarName"]
        for key in name_keys:
            result = self.__getitem__(key)
            if result != "unknown":
                break
        return result

    def release(self):
        """
        Free up the memory used by this message
        """
        eccodes.codes_release(self.gid)
        self.cache.clear()
        self.gid = None

    def get_dimension(self, dimensions=None, dimension_names=None):
        """
        get the shape of one message depending on the grid type

        Returns
        -------
        tuple
                (shape, dim-names)
        """
        if self["gridType"] == "rotated_ll":
            shape = (self["Nj"], self["Ni"])
            dim_names = ["rlat", "rlon"]
        elif self["gridType"] ==  "regular_ll":
            shape = (self["Nj"], self["Ni"])
            dim_names = ["lat", "lon"]
        elif self["gridType"] in ["sh", "reduced_gg", "unstructured_grid"]:
            shape = (self["numberOfValues"],)
            dim_names = ["ncells"]
        else:
            raise ValueError("don't know how to calculate the shape for grid type %s" % self["gridType"])

        # loop over all already used dims for comparison
        if dimensions is not None and dimension_names is not None:
            for one_var in dimensions.keys():
                if dimension_names[one_var] == dim_names and dimensions[one_var] != shape:
                    for id, dn in enumerate(dim_names):
                        dim_names[id] = "%s%d" % (dn, 2)
        return shape, dim_names

    def get_coordinates(self, dimension_names):
        """
        get the longitude and latitude coordinates for one message

        Returns
        -------
        tuple:
            ((lon-dim-names, lon-coord), (lat-dim-names), lat-coord)
        """
        # are coordinates available?
        if "longitudes" not in self or "latitudes" not in self:
            return None, None

        if self["gridType"] == "rotated_ll":
            lon = (dimension_names, numpy.array(self["longitudes"].reshape(self["Nj"], self["Ni"]), dtype=numpy.float32))
            lat = (dimension_names, numpy.array(self["latitudes"].reshape(self["Nj"], self["Ni"]), dtype=numpy.float32))
        elif self["gridType"] in ["sh", "reduced_gg", "unstructured_grid"]:
            lon = (dimension_names[0], numpy.array(self["longitudes"], dtype=numpy.float32))
            lat = (dimension_names[0], numpy.array(self["latitudes"], dtype=numpy.float32))
        elif self["gridType"] == "regular_ll":
            lon = (dimension_names[1], numpy.array(self["longitudes"].reshape(self["Nj"], self["Ni"])[0, :], dtype=numpy.float32))
            lat = (dimension_names[0], numpy.array(self["latitudes"].reshape(self["Nj"], self["Ni"])[:, 0], dtype=numpy.float32))
        else:
            lon = (dimension_names[1], numpy.array(self["longitudes"], dtype=numpy.float32))
            lat = (dimension_names[0], numpy.array(self["latitudes"], dtype=numpy.float32))
        return lon, lat

    def get_rotated_ll_info(self, dim_names):
        """
        get the rotated pole and the rotated lon/lat coordinates

        Parameters
        ----------
        dim_names : list
                names of the rlat and rlon dimensions

        Returns
        -------

        """
        if self["gridType"] != "rotated_ll":
            raise ValueError("The gridType '%s' has not rotated pole!" % self["gridType"])
        rotated_pole_name = "rotated_pole"
        if not dim_names[0].endswith("t"):
            rotated_pole_name += dim_names[0][-1]
        # create rotated pole description
        rotated_pole = xarray.DataArray(numpy.zeros(1, dtype=numpy.int8), dims=(rotated_pole_name,))
        rotated_pole.attrs["grid_mapping_name"] = "rotated_latitude_longitude"
        rotated_pole.attrs["grid_north_pole_latitude"] = self["latitudeOfSouthernPoleInDegrees"] * -1
        rotated_pole.attrs["grid_north_pole_longitude"] = self["longitudeOfSouthernPoleInDegrees"] - 180
        # create rotated coordinate arrays
        first_lon = self["longitudeOfFirstGridPointInDegrees"]
        last_lon = self["longitudeOfLastGridPointInDegrees"]
        if last_lon < first_lon and first_lon > 180:
            first_lon -= 360
        rlon = xarray.DataArray(numpy.arange(first_lon,
                                             last_lon + self["iDirectionIncrementInDegrees"],
                                             self["iDirectionIncrementInDegrees"], dtype=numpy.float32),
                                dims=(dim_names[-1],))
        rlon.attrs["long_name"] = "longitude in rotated pole grid"
        rlon.attrs["units"] = "degrees"
        rlon.attrs["standard_name"] = "grid_longitude"
        rlat = xarray.DataArray(numpy.arange(self["latitudeOfFirstGridPointInDegrees"],
                                             self["latitudeOfLastGridPointInDegrees"] + self["jDirectionIncrementInDegrees"],
                                             self["jDirectionIncrementInDegrees"], dtype=numpy.float32),
                                dims=(dim_names[-2],))
        rlat.attrs["long_name"] = "latitude in rotated pole grid"
        rlat.attrs["units"] = "degrees"
        rlat.attrs["standard_name"] = "grid_latitude"
        return rotated_pole_name, rotated_pole, rlat, rlon

    def get_level(self):
        """
        gets the center value of the level coordinate, or if available first and second layer

        """
        if self["typeOfLevel"] in ["generalVerticalLayer", "isobaricInhPa"]:
            return self["level"]
        if not "scaledValueOfFirstFixedSurface" in self or not "scaledValueOfSecondFixedSurface" in self:
            return self["level"]
        first_surface = self["scaledValueOfFirstFixedSurface"]
        second_surface = self["scaledValueOfSecondFixedSurface"]
        first_missing = first_surface == eccodes.CODES_MISSING_LONG or first_surface == eccodes.CODES_MISSING_DOUBLE
        second_missing = second_surface == eccodes.CODES_MISSING_LONG or second_surface == eccodes.CODES_MISSING_DOUBLE

        if first_missing and not second_missing:
            return second_surface
        elif not first_missing and second_missing:
            return first_surface
        elif first_missing and second_missing:
            return self["level"]
        else:
            return first_surface, second_surface
