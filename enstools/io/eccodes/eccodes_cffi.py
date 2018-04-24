"""
A minial Interface to ecCodes based on CFFI
"""
import cffi
import numpy as np
import xarray

# initialize the interface to the C-Library
ffi = cffi.FFI()

# type definition. No need to specify internals of the structs. just name them
ffi.cdef("typedef struct codes_handle codes_handle;")
ffi.cdef("typedef struct codes_context codes_context;")
ffi.cdef("typedef struct codes_keys_iterator codes_keys_iterator;")

# definition of the used functions
ffi.cdef("long codes_get_api_version (void);")
ffi.cdef("codes_handle* codes_handle_new_from_message(codes_context* c, const void*	data, size_t data_len);")
ffi.cdef("int codes_handle_delete(codes_handle* h);")
ffi.cdef("int codes_get_long(codes_handle* h, const char* key, long* value);")
ffi.cdef("int codes_get_double(codes_handle* h, const char* key, double* value);")
ffi.cdef("int codes_get_string(codes_handle* h, const char* key, char* mesg, size_t* length);")
ffi.cdef("int codes_get_size(codes_handle* h, const char* key, size_t* size);")
ffi.cdef("int codes_get_long_array(codes_handle* h, const char* key, long* vals, size_t* length);")
ffi.cdef("int codes_get_double_array(codes_handle* h, const char* key, double* vals, size_t* length);")
ffi.cdef("int grib_get_native_type(codes_handle* h, const char* name, int* type);")

# functions for key-iterators
ffi.cdef("codes_keys_iterator* codes_keys_iterator_new(codes_handle *h, unsigned long filter_flags, const char* name_space);")
ffi.cdef("int codes_keys_iterator_next (codes_keys_iterator *kiter);")
ffi.cdef("const char* codes_keys_iterator_get_name(codes_keys_iterator *kiter);")
ffi.cdef("int codes_keys_iterator_delete(codes_keys_iterator *kiter);")

# load the actual c-library
_eccodes = ffi.dlopen("libeccodes.so")

# Constants for 'missing'
CODES_MISSING_DOUBLE = -1e+100
CODES_MISSING_LONG = 2147483647


# A representation of one grib message
class GribMessage():

    def __init__(self, data):
        """
        create a message from the data buffer object

        Parameters
        ----------
        data
        """
        # create the grib message handle
        self.buffer = data
        self.handle = _eccodes.codes_handle_new_from_message(ffi.NULL, ffi.from_buffer(self.buffer), len(self.buffer))
        if self.handle == ffi.NULL:
            raise ValueError("unable to read grib message from buffer!")

        # read all cache
        self.cache = {}

    def __getitem__(self, item):
        if item in self.cache:
            return self.cache[item]
        else:
            try:
                ckey = _cstr(item)
                nelements = self.__codes_get_size(ckey)
                if nelements > 1:
                    value = self.__codes_get_array(ckey, nelements)
                else:
                    value = self.__codes_get(ckey)
                self.cache[item] = value
            except ValueError:
                # nothing found? Any error is interpreted as Key not found.
                raise KeyError("key '%s' not found in grib message!" % item)
            return value

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
        elif self["gridType"] == "regular_ll":
            shape = (self["Nj"], self["Ni"])
            dim_names = ["lat", "lon"]
        elif self["gridType"] in ["sh", "reduced_gg", "unstructured_grid"]:
            shape = (self["numberOfValues"],)
            dim_names = ["cell"]
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
            lon = (dimension_names, np.array(self["longitudes"].reshape(self["Nj"], self["Ni"]), dtype=np.float32))
            lat = (dimension_names, np.array(self["latitudes"].reshape(self["Nj"], self["Ni"]), dtype=np.float32))
        elif self["gridType"] in ["sh", "reduced_gg", "unstructured_grid"]:
            lon = (dimension_names[0], np.array(self["longitudes"], dtype=np.float32))
            lat = (dimension_names[0], np.array(self["latitudes"], dtype=np.float32))
        elif self["gridType"] == "regular_ll":
            lon = (dimension_names[1], np.array(self["longitudes"].reshape(self["Nj"], self["Ni"])[0, :], dtype=np.float32))
            lat = (dimension_names[0], np.array(self["latitudes"].reshape(self["Nj"], self["Ni"])[:, 0], dtype=np.float32))
        else:
            lon = (dimension_names[1], np.array(self["longitudes"], dtype=np.float32))
            lat = (dimension_names[0], np.array(self["latitudes"], dtype=np.float32))
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
        rotated_pole = xarray.DataArray(np.zeros(1, dtype=np.int8), dims=(rotated_pole_name,))
        rotated_pole.attrs["grid_mapping_name"] = "rotated_latitude_longitude"
        rotated_pole.attrs["grid_north_pole_latitude"] = self["latitudeOfSouthernPoleInDegrees"] * -1
        rotated_pole.attrs["grid_north_pole_longitude"] = self["longitudeOfSouthernPoleInDegrees"] - 180
        # create rotated coordinate arrays
        first_lon = self["longitudeOfFirstGridPointInDegrees"]
        last_lon = self["longitudeOfLastGridPointInDegrees"]
        if last_lon < first_lon and first_lon > 180:
            first_lon -= 360
        rlon = xarray.DataArray(np.arange(first_lon,
                                             last_lon + self["iDirectionIncrementInDegrees"],
                                             self["iDirectionIncrementInDegrees"], dtype=np.float32),
                                dims=(dim_names[-1],))
        rlon.attrs["long_name"] = "longitude in rotated pole grid"
        rlon.attrs["units"] = "degrees"
        rlon.attrs["standard_name"] = "grid_longitude"
        rlat = xarray.DataArray(np.arange(self["latitudeOfFirstGridPointInDegrees"],
                                             self["latitudeOfLastGridPointInDegrees"] + self["jDirectionIncrementInDegrees"],
                                             self["jDirectionIncrementInDegrees"], dtype=np.float32),
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
        first_missing = first_surface == CODES_MISSING_LONG or first_surface == CODES_MISSING_DOUBLE
        second_missing = second_surface == CODES_MISSING_LONG or second_surface == CODES_MISSING_DOUBLE

        if first_missing and not second_missing:
            return second_surface
        elif not first_missing and second_missing:
            return first_surface
        elif first_missing and second_missing:
            return self["level"]
        else:
            return first_surface, second_surface

    def get_values(self, shape=None, dtype=None, missing=None):
        """
        read the encoded values from the message

        Parameters
        ----------
        dtype : np.dtype
                values are returned in an array of the specified type

        missing : float
                value used within the grib message the mark missing values. The returned array will contain NaN at this
                locations.

        Returns
        -------
        np.ndarray
        """
        values = self["values"]
        if shape is not None:
            values = values.reshape(shape)
        if dtype is not None and dtype != np.float64:
            values = np.array(values, dtype=dtype)
        # replace fill values with nan
        values = np.where(values == missing, np.nan, values)
        return values

    def __grib_get_native_type(self, key):
        """
        Get the native type of a specific grib key
        """
        itype = ffi.new("int[1]")
        err = _eccodes.grib_get_native_type(self.handle, key, itype)
        if err != 0:
            raise ValueError("unable to get type of key '%s'" % ffi.string(key))
        if itype[0] == 1:
            return int
        elif itype[0] == 2:
            return float
        else:
            return str

    def __codes_get_size(self, key):
        """
        get the number of elements for a given key

        Parameters
        ----------
        key : cstr
                name of the key

        Returns
        -------
        int :
                number of elements
        """
        size = ffi.new("size_t[1]")
        err = _eccodes.codes_get_size(self.handle, key, size)
        if err != 0:
            raise ValueError("unable to get number of elements for key '%s'" % ffi.string(key))
        return size[0]

    def __codes_get(self, key):
        """
        get the value of a non-array key

        Parameters
        ----------
        key : cstr
                name of the key

        Returns
        -------
        int or float or str
        """
        key_type = self.__grib_get_native_type(key)
        if key_type == int:
            value_ptr = ffi.new("long[1]")
            err = _eccodes.codes_get_long(self.handle, key, value_ptr)
            value = value_ptr[0]
        elif key_type == float:
            value_ptr = ffi.new("double[1]")
            err = _eccodes.codes_get_double(self.handle, key, value_ptr)
            value = value_ptr[0]
        else:
            value_buffer = np.zeros(1024, dtype=np.uint8)
            value_buffer_length = ffi.new("size_t[1]", init=[1024])
            err = _eccodes.codes_get_string(self.handle, key, ffi.from_buffer(value_buffer), value_buffer_length)
            if value_buffer_length[0] == 1024:
                value_buffer_length[0] = np.where(value_buffer == 0)[0][0]
            value = str(value_buffer[:value_buffer_length[0]-1].tostring(), "utf-8")
        if err != 0:
            raise ValueError("unable to get value for key '%s'" % ffi.string(key))
        return value

    def __codes_get_array(self, key, nelements):
        """
        Get a values for a key with multiple values

        Parameters
        ----------
        key : cstr
                name of the key

        nelements : int
                size the array to retrieve

        Returns
        -------
        np.ndarray
        """
        key_type = self.__grib_get_native_type(key)
        length = ffi.new("size_t[1]")
        length[0] = nelements
        if key_type == int:
            values = np.empty(nelements, dtype=np.int64)
            err = _eccodes.codes_get_long_array(self.handle, key, ffi.from_buffer(values), length)
        elif key_type == float:
            values = np.empty(nelements, dtype=np.float64)
            err = _eccodes.codes_get_double_array(self.handle, key, ffi.from_buffer(values), length)
        else:
            raise ValueError("string arrays are not yet supported!")
        if err != 0:
            raise ValueError("unable to get value for key '%s'" % ffi.string(key))
        return values

    def __del__(self):
        """
        free up the memory
        """
        err = _eccodes.codes_handle_delete(self.handle)
        if err != 0:
            raise ValueError("unable to free memory of grib message!")


def _cstr(pstr):
    """
    convert a python string object into a c string object (copy).

    Parameters
    ----------
    pstr : str
            python string

    Returns
    -------
    const char*
    """
    buffer = np.fromstring(pstr + "\x00", dtype=np.uint8)
    result = ffi.from_buffer(buffer)
    return result


def read_message_header_bytes(infile, offset, read_data=False):
    """
    Read the header of a grib message and return an byte array with the length of the full message, but without
    the actual data

    Parameters
    ----------
    infile

    Returns
    -------

    """
    # find the start word GRIB
    infile.seek(offset)
    start = infile.read(20)
    istart = start.find(b"GRIB")
    if istart == -1:
        return None
    offset += istart

    # find at first the grib edition to account for different formats
    infile.seek(offset + 7)
    edition = int(infile.read(1)[0])

    # get the length of the total message
    if edition == 1:
        # read the first section
        infile.seek(offset)
        section0 = infile.read(8)
        length_total = int.from_bytes(section0[4:7], byteorder='big')

        # create an numpy array with the total size of the message
        bytes = np.zeros(length_total, dtype=np.uint8)  # bytearray(length_total)
        bytes[0:8] = np.fromstring(section0, dtype=np.uint8)
        pos = 8

        # read the complete message?
        if read_data:
            bytes[8:] = np.fromstring(infile.read(length_total - 8), dtype=np.uint8)
            return bytes

        # read the first sections, but not the data
        for sec in range(1, 5):
            # read the length of the section
            bytes[pos:pos+3] = np.fromstring(infile.read(3), dtype=np.uint8)
            length_sec = int.from_bytes(bytes[pos:pos+3], byteorder='big')

            # do not read if this is the final data section
            if pos + length_sec + 4 == length_total:
                # read the first bytes only
                bytes[pos+3:pos+11] = np.fromstring(infile.read(8), dtype=np.uint8)
                infile.seek(offset + length_total - 5)
                bytes[-5:] = np.fromstring(infile.read(5), dtype=np.uint8)
                break
            else:
                # read data of this section
                bytes[pos+3:pos+length_sec] = np.fromstring(infile.read(length_sec - 3), dtype=np.uint8)
                pos = pos + length_sec
    else:
        raise ValueError("GRIB2 is not yet supported!")

    return bytes

    #length_total =
    #edition =
    #f.seek(istart + length_total - 4)
    #bytes = f.read(4)
    #print(bytes)
    #print(istart, length_total, edition)
