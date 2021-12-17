from typing import List, Union
from .availability_checks import check_sz_availability


#############################################
# SZ Specific functions
def parse_sz_compression_options(arguments: List[str]):
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


def sz_encoding(compression_options: List[str]) -> Union[int, tuple]:
    """
    Create a dictionary with the encoding that will be passed to the hdf5 engine, to use the ZFP filter.
    One and only of the method options need to be provided: rate, precision or accuracy.
    Parameters
    ----------
    compression_options: list of strings

    :returns: sz_filter_id, compression_opts
    :rtype: int, dict
    """
    assert check_sz_availability(), "Attempting to use SZ filter which is not available."
    # The unique filter id given by HDF5
    sz_filter_id = 32017

    # Check options provided:
    compressor, method, parameter = compression_options
    assert compressor == "sz", "Passing wrong options"
    # Get ZFP encoding options
    if method == "abs":
        sz_mode = 0
    elif method == "rel":
        sz_mode = 1
    elif method == "pw_rel":
        sz_mode = 10
    else:
        raise NotImplementedError("Method %s has not been implemented yet" % method)
    compression_opts = (sz_mode, 0, sz_pack_error(parameter))
    return sz_filter_id, compression_opts


def sz_pack_error(error: float) -> int:
    from struct import pack, unpack
    return unpack('I', pack('<f', error))[0]  # Pack as IEEE 754 single


