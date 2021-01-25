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
        # Blosc encoding
        compressor = "lz4"
        clevel = 9
        encoding = {}

        BLOSC_filter_id = 32001
        variables = [var for var in ds.variables]

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
            #encoding[variable]["chunksizes"] = dataset[variable].chunks
            #encoding[variable]["original_shape"] = dataset[variable].shape
        return encoding
    elif mode == "lossy":
                # Blosc encoding
        encoding = {}

        ZFP_filter_id = 32013
        # ZFP modes
        ZFP_MODE_RATE = 1
        ZFP_MODE_PRECISION = 2
        ZFP_MODE_ACCURACY = 3
        ZFP_MODE_EXPERT = 4

        variables = [var for var in ds.variables]

        def zfp_rate_opts(rate):
            from struct import pack, unpack
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

        
        # Define the compression_opts array that will be passed to the filter
        compression_opts = zfp_rate_opts(4)

        #Set the enconding for each variable
        for variable in variables:
            encoding[variable] = {}
            encoding[variable]["compression"] = ZFP_filter_id
            encoding[variable]["compression_opts"] = compression_opts
            """
            # This was working with the hdf5 tests and gives problems with grib data from DWD operational.
            try:
                t,z,x,y = ds[variable].shape
                encoding[variable]["chunksizes"] = (t,1,x,y)
            except TypeError:
                encoding[variable]["chunksizes"] = ds[variable].chunks
            except ValueError:
                encoding[variable]["chunksizes"] = ds[variable].chunks
            """
            #encoding[variable]["chunksizes"] = dataset[variable].chunks
            #encoding[variable]["original_shape"] = dataset[variable].shape
        return encoding
    else:
        return None
