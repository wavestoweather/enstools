def set_encoding(ds, compression_options):
    """
    Create a dictionary with the encoding that will be passed to the hdf5 engine.

    Parameters
    ----------
    ds : xarray.Dataset
            the dataset that will be stored

    compression_options: string or filepath
        It can contain the string that defines the compression options that will be used in the whole dataset,
        or a filepath to a configuration file in which we might have per variable specification.
        For lossless compression, we can choose the backend and the compression leven as follows
            "lossless:backend:compression_level(from 1 to 9)"
        The backend must be one of the following options:
                'blosclz'
                'lz4'
                'lz4hc'
                'snappy'
                'zlib'
                'zstd'
        For lossy compression, we can choose the compressor (zfp or sz),
        the method and the method parameter (the accuracy, the precision or the rate).
        Some examples:
            "lossless"
            "lossy"
            "lossless:zlib:5"
            "lossy:zfp:accuracy:0.00001"
            "lossy:zfp:precision:12"
            "lossy:sz:abs:0.1"
            "lossy:sz:rel:0.01"
            
        If using a configuration file, the file should follow a json format and can contain per-variable values.
        It is also possible to define a default option. For example:
        { "default": "lossless",
          "temp": "lossy:zfp:accuracy:0.1",
          "qv": "lossy:zfp:accuracy:0.00001"        
        }
            
           
    
    """
    # Parsing the compression options
    mode, options = parse_compression_options(compression_options)

    # Get list of variable names and coordinates
    variables = [var for var in ds.variables]
    coordinates = [v for v in ds.coords]
    data_variables = [v for v in ds.variables if v not in coordinates]

    # Initialize encoding dictionary
    encoding = {}

    _, lossless_options = parse_compression_options("lossless")
    # Defining lossless parameters
    lossless_compressor, lossless_clevel = lossless_options
    lossless_filter_id, lossless_compression_options = blosc_encoding(compressor=lossless_compressor,
                                                                      clevel=lossless_clevel)

    # If using a single mode for all the variables, find the corresponding filter_id
    # and options and fill the encoding dictionary with the information.
    if mode is None:
        return None
    elif mode == "lossless":
        for variable in variables:
            encoding[variable] = {}
            encoding[variable]["compression"] = lossless_filter_id
            encoding[variable]["compression_opts"] = lossless_compression_options

    elif mode == "lossy":
        lossy_filter_id, lossy_compression_options = lossy_encoding(options)

        for variable in variables:
            encoding[variable] = {}
            if len(ds[variable].shape) > 1 and variable in data_variables:
                    encoding[variable]["compression"] = lossy_filter_id
                    # In some cases (i.e. SZ) the compression options need to be adapted to acount for data dimensions
                    variable_compression_options = adapt_compression_options(
                        lossy_filter_id,
                        lossy_compression_options,
                        ds[variable])
                    encoding[variable]["compression_opts"] = variable_compression_options
                    encoding[variable]["chunksizes"] = ds[variable].shape
            else:
                encoding[variable]["compression"] = lossless_filter_id
                encoding[variable]["compression_opts"] = lossless_compression_options

    # In the case of using a configuration file , we might have a different encoding specification for each variable    
    elif mode == "file":
        filename = options[0]
        # Read the file to get the per variable specifications
        dictionary_of_filter_ids, dictionary_of_compression_options = parse_configuration_file(filename)

        # Check if there's a default defined, otherwise use BLOSC encoding with default arguments
        try:
            default_id = dictionary_of_filter_ids["default"]
            default_options = dictionary_of_compression_options["default"]
        except KeyError:
            default_id, default_options = blosc_encoding()

        # Loop through variables and use the custom parameters if exist and otherwise the default ones
        for variable in variables:
            # Use default parameters for coordinates
            if variable in coordinates:
                filter_id = default_id
                compression_options = default_options
            elif len(ds[variable].shape) < 3:
                filter_id = default_id
                compression_options = default_options
            else:
                try:
                    filter_id = dictionary_of_filter_ids[variable]
                    compression_options = dictionary_of_compression_options[variable]
                except KeyError:
                    filter_id = default_id
                    compression_options = default_options

            # Fill the encoding dictionary
            encoding[variable] = {}
            encoding[variable]["compression"] = filter_id
            variable_compression_options = adapt_compression_options(filter_id, compression_options, ds[variable])
            encoding[variable]["compression_opts"] = variable_compression_options
            encoding[variable]["chunksizes"] = ds[variable].shape

    else:
        return None

    return encoding


def adapt_compression_options(filter_id, compression_options, variable_dataset):
    if filter_id == 32013:
        return compression_options
    elif filter_id == 32017:
        # Case SZ (filter id 32017)
        # Compression_opts have to include dimensions of data and shape.
        shape = variable_dataset.shape
        dim = len(shape)
        data_type = 0  # TODO: Maybe we can set this dynamically instead of having it hardcoded here.
        error_mode = compression_options[0]
        error_val = compression_options[2]
        return (dim, data_type, *shape, error_mode, error_val, error_val, error_val)
    else:
        return compression_options


def filter_and_options_from_command_line_arguments(string):
    """
    Given a configuration string , it will return the proper filter_id and compressor options.

    Parameters
    ----------
    string:  string
           compression configuration string (i.e.  "lossy:zfp:accuracy:0.1")
    """
    # Parsing the compression options
    mode, options = parse_compression_options(string)
    assert mode != "file", "Compression: File method argument should not happen here"
    if mode == "lossless":
        compressor, clevel = options
        return blosc_encoding(compressor=compressor, clevel=clevel)
    elif mode == "lossy":
        return lossy_encoding(options)


def blosc_encoding(compressor="lz4", clevel=9):
    """
    The function will return the BLOSC_is and the compression options .

    Parameters
    ----------
    compressor:  string
            the backend that will be used in BLOSC. The avaliable options are:
                'blosclz'
                'lz4'
                'lz4hc'
                'snappy'
                'zlib'
                'zstd'
    clevel: integer
        Compression level From 1 to 9
    """
    # The uniq filter id given by HDF5
    BLOSC_filter_id = 32001
    # For now, the shuffle its always activated
    shuffle = 1

    # Available backends
    compressors = {
        'blosclz': 0,
        'lz4': 1,
        'lz4hc': 2,
        'snappy': 3,
        'zlib': 4,
        'zstd': 5,
    }
    # Get the compressor id from the compressors dictionary

    compressor_id = compressors[compressor]

    # Define the compression_opts array that will be passed to the filter
    compression_opts = (0, 0, 0, 0, clevel, shuffle, compressor_id)

    return BLOSC_filter_id, compression_opts


def lossy_encoding(compression_options):
    if compression_options[0] == "zfp":
        return zfp_encoding(compression_options)
    elif compression_options[0] == "sz":
        return sz_encoding(compression_options)
    else:
        raise NotImplementedError


def zfp_encoding(compression_options):
    """
    Create a dictionary with the encoding that will be passed to the hdf5 engine, to use the ZFP filter.
    One and only of the method options need to be provided: rate, precision or accuracy.
    Parameters
    ----------
    compression_options: list of strings
    """
    # The uniq filter id given by HDF5 
    ZFP_filter_id = 32013

    # Check options provided:
    compressor, method, parameter = compression_options
    assert compressor == "zfp", "Passing wrong options"
    # Get ZFP encoding options
    if method == "rate":
        compression_opts = zfp_rate_opts(parameter)
    elif method == "precision":
        compression_opts = zfp_precision_opts(parameter)
    elif method == "accuracy":
        compression_opts = zfp_accuracy_opts(parameter)
    elif method == "reversible":
        compression_opts = zfp_reversible()
    else:
        raise NotImplementedError("Method %s has not been implemented yet" % method)

    return ZFP_filter_id, compression_opts


def sz_encoding(compression_options):
    """
    Create a dictionary with the encoding that will be passed to the hdf5 engine, to use the ZFP filter.
    One and only of the method options need to be provided: rate, precision or accuracy.
    Parameters
    ----------
    compression_options: list of strings
    """
    # The uniq filter id given by HDF5
    sz_filter_id = 32017

    # Check options provided:
    compressor, method, parameter = compression_options
    assert compressor == "sz", "Passing wrong options"
    # Get ZFP encoding options
    if method == "abs":
        SZ_MODE = 0
    elif method == "rel":
        SZ_MODE = 1
    elif method == "pw_rel":
        SZ_MODE = 2
    else:
        raise NotImplementedError("Method %s has not been implemented yet" % method)
    compression_opts = (SZ_MODE, 0, sz_pack_error(parameter))
    return sz_filter_id, compression_opts


def sz_pack_error(error):
    from struct import pack, unpack
    return unpack('I', pack('<f', error))[0]  # Pack as IEEE 754 single


# Some functions to define the compression_opts array that will be passed to the filter
def zfp_rate_opts(rate):
    """Create compression options for ZFP in fixed-rate mode

    The float rate parameter is the number of compressed bits per value.
    """
    ZFP_MODE_RATE = 1
    from struct import pack, unpack
    rate = pack('<d', rate)  # Pack as IEEE 754 double
    high = unpack('<I', rate[0:4])[0]  # Unpack high bits as unsigned int
    low = unpack('<I', rate[4:8])[0]  # Unpack low bits as unsigned int
    return ZFP_MODE_RATE, 0, high, low, 0, 0


def zfp_precision_opts(precision):
    """Create a compression options for ZFP in fixed-precision mode

    The float precision parameter is the number of uncompressed bits per value.
    """
    ZFP_MODE_PRECISION = 2
    return ZFP_MODE_PRECISION, 0, precision, 0, 0, 0


def zfp_accuracy_opts(accuracy):
    """Create compression options for ZFP in fixed-accuracy mode

    The float accuracy parameter is the absolute error tolarance (e.g. 0.001).
    """
    ZFP_MODE_ACCURACY = 3
    from struct import pack, unpack
    accuracy = pack('<d', accuracy)  # Pack as IEEE 754 double
    high = unpack('<I', accuracy[0:4])[0]  # Unpack high bits as unsigned int
    low = unpack('<I', accuracy[4:8])[0]  # Unpack low bits as unsigned int
    return ZFP_MODE_ACCURACY, 0, high, low, 0, 0


def zfp_expert_opts(minbits, maxbits, maxprec, minexp):
    """Create compression options for ZFP in "expert" mode

    See the ZFP docs for the meaning of the parameters.
    """
    ZFP_MODE_EXPERT = 4
    return ZFP_MODE_EXPERT, 0, minbits, maxbits, maxprec, minexp


def zfp_reversible():
    """Create compression options for ZFP in reversible mode

    It should result in lossless compression.
    """
    ZFP_MODE_REVERSIBLE = 5
    return ZFP_MODE_REVERSIBLE, 0, 0, 0, 0, 0


# Functions to parse the compression options,
# to give more flexibility and allow a way to pass custom compression options
# TODO: Move to a different file?
def parse_compression_options(string):
    from os.path import isfile
    """
    Function to parse compression options
    Input
    -----
    string: string
    
    Output
    -----
    compression_method : string
                        "lossy", "lossless" or None
    compression_opts:  tuple
                       for lossless: (backend, clevel) 
                       for lossy:    (backend, method, parameter)
    """
    arguments = string.split(":")
    # Check that all arguments have information
    assert not any(
        [not argument.strip() for argument in arguments]), "Compression: The provided option has a wrong format."
    first = arguments[0]
    # Check if it is a file:
    if isfile(first):
        return "file", [first]
    elif first == "lossless":
        return parse_lossless_compression_options(arguments)
    elif first == "lossy":
        return parse_lossy_compression_options(arguments)
    elif first == "None":
        return None, None
    else:
        raise AssertionError("Compression: The argument should be lossless/lossy/or a path to a file.")


def parse_lossless_compression_options(arguments):
    """
    Function to parse compression options for the lossless case
    Input
    -----
    arguments: list of strings
    
    Output
    -----
    compression_method : "lossless"
    compression_opts:  tuple
                       (backend:string, clevel:int) 
    """

    BLOSC_backends = ['blosclz',
                      'lz4',
                      'lz4hc',
                      'snappy',
                      'zlib',
                      'zstd']
    if len(arguments) == 1:
        # By default, lossless compression will use lz4 backend and the maximum compression level
        backend = "lz4"
        compression_level = 9
    elif len(arguments) == 2:
        assert arguments[1] in BLOSC_backends, "Unknwown backend %s" % arguments[1]
        # In case the backend its selected but the compression level its not specified,
        # the intermediate level 5 will be selected
        backend = arguments[1]
        compression_level = 5
    elif len(arguments) == 3:
        backend = arguments[1]
        try:
            compression_level = int(arguments[2])
        except ValueError:
            raise AssertionError(
                "Compression: Invalid value '%s' for compression level. Must be a value between 1 and 9" % arguments[2])
        assert 1 <= compression_level <= 9,\
            "Compression: Invalid value '%s' for compression level. Must be a value between 1 and 9" % arguments[2]
    else:
        raise AssertionError("Compression: Wrong number of arguments in %s" % ":".join(arguments))
    return "lossless", (backend, compression_level)


def parse_lossy_compression_options(arguments):
    """
    Function to parse compression options for the lossy case
    Input
    -----
    arguments: list of strings
    
    Output
    -----
    compression_method : string="lossy"
    compression_opts:  tuple
                       (backend:string, method:string, parameter:int or float)
    """

    if len(arguments) == 1:
        return "lossy", ("zfp", "rate", 4)
    else:
        # Right now we only have the zfp case:
        if arguments[1] == "zfp":
            return parse_zfp_compression_options(arguments)
        elif arguments[1] == "sz":
            return parse_sz_compression_options(arguments)
        else:
            raise AssertionError("Compression: Unknown lossy compressor: %s" % arguments[1])


def parse_zfp_compression_options(arguments):
    """
    Function to parse compression options for the ZFP compressor
    Input
    -----
    arguments: list of strings
    
    Output
    -----
    compression_method : string="lossy"
    compression_opts:  tuple
                       (backend:string, method:string, parameter:int or float)
    """
    if len(arguments) == 2:
        return "lossy", ("zfp", "rate", 6)
    assert len(arguments) == 4, "Compression: ZFP compression requires 4 arguments: lossy:zfp:method:value"
    try:
        if arguments[2] == "rate":
            rate = int(arguments[3])
            return "lossy", ("zfp", "rate", rate)
        elif arguments[2] == "precision":
            precision = int(arguments[3])
            return "lossy", ("zfp", "precision", precision)
        elif arguments[2] == "accuracy":
            accuracy = float(arguments[3])
            return "lossy", ("zfp", "accuracy", accuracy)
        else:
            raise AssertionError("Compression: Unknown ZFP method.")
    except ValueError:
        raise AssertionError("Compression: Invalid value '%s' for ZFP" % arguments[3])


def parse_sz_compression_options(arguments):
    """
    Function to parse compression options for the SZ compressor
    Input
    -----
    arguments: list of strings

    Output
    -----
    compression_method : string="lossy"
    compression_opts:  tuple
                       (backend:string, method:string, parameter:int or float)
    """
    default = ("lossy", ("sz", "pw_rel", .01))
    if len(arguments) == 2:
        return default
    assert len(arguments) == 4, "Compression: SZ compression requires 4 arguments: lossy:zfp:method:value"
    try:
        if arguments[2] == "abs":
            abs_error = float(arguments[3])
            return "lossy", ("sz", "abs", abs_error)
        elif arguments[2] == "rel":
            rel_error = float(arguments[3])
            return "lossy", ("sz", "rel", rel_error)
        elif arguments[2] == "pw_rel":
            pw_rel_error = float(arguments[3])
            return "lossy", ("sz", "pw_rel", pw_rel_error)
        else:
            raise AssertionError("Compression: Unknown SZ method.")
    except ValueError:
        raise AssertionError("Compression: Invalid value '%s' for SZ" % arguments[3])


def parse_configuration_file(filename):
    import json

    # Initialize dictionaries
    dictionary_of_filter_ids = {}
    dictionary_of_compression_options = {}

    with open(filename) as f:
        specifications = json.loads(f.read())

    for key, options in specifications.items():
        filter_id, compression_options = filter_and_options_from_command_line_arguments(options)
        dictionary_of_filter_ids[key] = filter_id
        dictionary_of_compression_options[key] = compression_options

    return dictionary_of_filter_ids, dictionary_of_compression_options


def encoding_description(encoding):
    """
    Function to create a meaningful description that will be added as variable attribute into the netcdf file.
    Input:
    ------

    encoding: dict
        Dictionary with the encoding specifications for each variable.
    """
    labels = {
        32001: "BLOSC",
        32013: "ZFP",
        32017: "SZ",
    }
    descriptions = {}
    for variable, var_encoding in encoding.items():
        try:
            filter_id = var_encoding["compression"]
            filter_name = labels[filter_id]
            if filter_name == "BLOSC":
                compression_type = "Lossless"
            else:
                compression_type = "Lossy"
            description = "Compressed using %s (id:%i) - %s" % (filter_name, filter_id, compression_type)
            descriptions[variable] = description
        except KeyError:
            description = "Non compressed"
            descriptions[variable] = description
    return descriptions
