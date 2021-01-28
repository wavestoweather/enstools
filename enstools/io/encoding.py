# Some definitions

# Some filter ids
BLOSC_filter_id = 32001
ZFP_filter_id = 32013

# ZFP modes
ZFP_MODE_RATE = 1
ZFP_MODE_PRECISION = 2
ZFP_MODE_ACCURACY = 3
ZFP_MODE_EXPERT = 4


def set_encoding(ds, compression_options):
    """
    Create a dictionary with the encoding that will be passed to the hdf5 engine.

    Parameters
    ----------
    ds : xarray.Dataset
            the dataset that will be stored

    mode: string
        "lossless"
        "lossy"
    
    """
    # Parsing the compression options
    mode, options = parse_compression_options(compression_options)
    
    if mode == "lossless":
        return BLOSC_encoding(ds,compressor=options[0],clevel=options[1])
    elif mode == "lossy":
        return ZFP_encoding(ds,options)
    else:
        return None

    
def BLOSC_encoding(dataset, compressor = "lz4", clevel = 9 ):
    """
    Create a dictionary with the encoding that will be passed to the hdf5 engine, to use the BLOSC filter.

    Parameters
    ----------
    ds : xarray.Dataset
            the dataset that will be stored

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
    encoding = {}

    
    variables = [var for var in dataset.variables]

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

    #Set the enconding for each variable
    for variable in variables:
        encoding[variable] = {}
        encoding[variable]["compression"] = BLOSC_filter_id
        encoding[variable]["compression_opts"] = compression_opts
    return encoding


def ZFP_encoding(ds, compression_options):
    """
    Create a dictionary with the encoding that will be passed to the hdf5 engine, to use the ZFP filter.
    One and only of the method options need to be provided: rate, precision or accuracy.
    Parameters
    ----------
    ds : xarray.Dataset
            the dataset that will be stored

    rate: integer
    
    precision: integer
    
    accuracy: float
    
    reversible: bool
    """
    # Check options provided:
    compressor, method, parameter = compression_options
    assert compressor == "zfp", "Passing wrong options"
    # ZFP encoding
    encoding = {}
    
    if method == "rate":
        compression_opts = zfp_rate_opts(parameter)
    elif method == "precision" :
        compression_opts = zfp_precision_opts(parameter)
    elif method == "accuracy" :
        compression_opts = zfp_accuracy_opts(parameter)
    elif method == "reversible":
        compression_opts = zfp_reversible()

    
    # Get list of variable names
    variables = [var for var in ds.variables]


    # Define the compression_opts array that will be passed to the filter


    #Set the enconding for each variable
    for variable in variables:
        encoding[variable] = {}
        encoding[variable]["compression"] = ZFP_filter_id
        encoding[variable]["compression_opts"] = compression_opts
    return encoding





# Some funcions to define the compression_opts array that will be passed to the filter
def zfp_rate_opts(rate):
    """Create compression options for ZFP in fixed-rate mode

    The float rate parameter is the number of compressed bits per value.
    """
    from struct import pack, unpack
    rate = pack('<d', rate)            # Pack as IEEE 754 double
    high = unpack('<I', rate[0:4])[0]  # Unpack high bits as unsigned int
    low = unpack('<I', rate[4:8])[0]   # Unpack low bits as unsigned int
    return (ZFP_MODE_RATE, 0, high, low, 0, 0)


def zfp_precision_opts(precision):
    """Create a compression options for ZFP in fixed-precision mode

    The float precision parameter is the number of uncompressed bits per value.
    """
    return (ZFP_MODE_PRECISION, 0, precision, 0, 0, 0)


def zfp_accuracy_opts(accuracy):
    """Create compression options for ZFP in fixed-accuracy mode

    The float accuracy parameter is the absolute error tolarance (e.g. 0.001).
    """
    from struct import pack, unpack
    accuracy = pack('<d', accuracy)        # Pack as IEEE 754 double
    high = unpack('<I', accuracy[0:4])[0]  # Unpack high bits as unsigned int
    low = unpack('<I', accuracy[4:8])[0]   # Unpack low bits as unsigned int
    return (ZFP_MODE_ACCURACY, 0, high, low, 0, 0)


def zfp_expert_opts(minbits, maxbits, maxprec, minexp):
    """Create compression options for ZFP in "expert" mode

    See the ZFP docs for the meaning of the parameters.
    """
    return (ZFP_MODE_EXPERT , 0, minbits, maxbits, maxprec, minexp)

def zfp_reversible():
    return (5, 0, 0, 0, 0, 0)


# Functions to parse the commpression options, to give more flexibility and allow a way to pass custom compression options
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
    arguments=string.split(":")
    # Check that all arguments have information
    assert not any([not argument.strip() for argument in arguments]), "Compression: The provided option has a wrong format."
    first = arguments[0]
    # Check if it is a file:
    if isfile(first):
        return parse_file(first)
    elif first == "lossless":
        return parse_lossless_compression_options(arguments)
    elif first == "lossy":
        return parse_lossy_compression_options(arguments)
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
        # In case the backend its selected but the compression level its not specified, the intermediate level 5 will be selected
        backend = arguments[1]
        compression_level = 5
    elif len(arguments) == 3:
        backend = arguments[1]
        try:
            compression_level = int(arguments[2])
        except ValueError:
            raise AssertionError("Compression: Invalid value '%s' for compression level. Must be a value between 1 and 9" % arguments[2])
        assert 1 <= compression_level <= 9, "Compression: Invalid value '%s' for compression level. Must be a value between 1 and 9" % arguments[2]
    else:
        raise AssertionError ("Compression: Wrong number of arguments in %s" % ":".join(arguments))
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
        # TODO: Check that this is the final desired behaviour. By default, the lossy compression will be set to zfp rate at 4 bits per value.
        return "lossy", ("zfp", "rate", 4)
    else:
        # Right now we only have the zfp case:
        if arguments[1] == "zfp":
            return parse_ZFP_compression_options(arguments)
        else:
            raise AssertionError("Compression: Unknown lossy compressor: %s" % arguments[1])

            
def parse_ZFP_compression_options(arguments):
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
        
def parse_file(arguments):
    raise NotImplementedError
