from typing import Tuple, List
from .availability_checks import check_blosc_availability


def parse_blosc_compression_options(arguments: List[str]):
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

    blosc_backends = ['blosclz',
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
        assert arguments[1] in blosc_backends, "Unknown backend %s" % arguments[1]
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
        assert 1 <= compression_level <= 9, \
            "Compression: Invalid value '%s' for compression level. Must be a value between 1 and 9" % arguments[2]
    else:
        raise AssertionError("Compression: Wrong number of arguments in %s" % ":".join(arguments))
    return "lossless", (backend, compression_level)


def blosc_encoding(compressor: str = "lz4", compressor_level: int = 9) -> Tuple[int, tuple]:
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
    compressor_level: integer
        Compression level From 1 to 9
    """
    assert check_blosc_availability(), "Attempting to use BLOSC filter which is not available."
    # The unique filter id given by HDF5
    blosc_filter_id = 32001
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
    compression_opts = (0, 0, 0, 0, compressor_level, shuffle, compressor_id)

    return blosc_filter_id, compression_opts
