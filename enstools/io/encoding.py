def set_encoding(ds, mode):
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
    if mode == "lossless":
        return BLOSC_encoding(ds,compressor="lz4",clevel=9)
    elif mode == "lossy":
        return ZFP_encoding(ds,rate=4)
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

    BLOSC_filter_id = 32001
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


def ZFP_encoding(ds, rate=None, precision=None, accuracy=None, reversible=None):
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
    # ZFP encoding
    encoding = {}
    
    
    if (rate is None) and (precision is None) and (accuracy is  None) and (reversible is None):
        raise AssertionError("One option needs to be provided.")
    if (rate is None) + (precision is None) + (accuracy is  None) + (reversible is None):
        raise AssertionError("Only one option needs to be provided.")
    
    # Get list of variable names
    variables = [var for var in ds.variables]


    # Define the compression_opts array that will be passed to the filter
    if rate is not None:
        compression_opts = zfp_rate_opts(rate)
    elif precision is not None:
        compression_opts = zfp_precision_opts(precision)
    elif accuracy is not None:
        compression_opts = zfp_accuracy_opts(accuracy)
    elif reversible is not None:
        compression_opts = zfp_reversible()

    #Set the enconding for each variable
    for variable in variables:
        encoding[variable] = {}
        encoding[variable]["compression"] = ZFP_filter_id
        encoding[variable]["compression_opts"] = compression_opts
    return encoding


ZFP_filter_id = 32013
# ZFP modes
ZFP_MODE_RATE = 1
ZFP_MODE_PRECISION = 2
ZFP_MODE_ACCURACY = 3
ZFP_MODE_EXPERT = 4
# Some funcions to define the compression_opts array that will be passed to the filter
def zfp_rate_opts(rate):
    """Create compression options for ZFP in fixed-rate mode

    The float rate parameter is the number of compressed bits per value.
    """
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