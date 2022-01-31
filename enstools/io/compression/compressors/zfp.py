from typing import List, Union
from .availability_checks import check_zfp_availability


####################################
# ZFP Specific functions
def parse_zfp_compression_options(arguments: List[str]):
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
            rate = float(arguments[3])
            return "lossy", ("zfp", "rate", rate)
        elif arguments[2] == "precision":
            precision = float(arguments[3])
            return "lossy", ("zfp", "precision", precision)
        elif arguments[2] == "accuracy":
            accuracy = float(arguments[3])
            return "lossy", ("zfp", "accuracy", accuracy)
        else:
            raise AssertionError("Compression: Unknown ZFP method.")
    except ValueError:
        raise AssertionError("Compression: Invalid value '%s' for ZFP" % arguments[3])


def zfp_encoding(compression_options: List[str]) -> Union[int, tuple]:
    """
    Create a dictionary with the encoding that will be passed to the hdf5 engine, to use the ZFP filter.
    One and only of the method options need to be provided: rate, precision or accuracy.
    Parameters
    ----------
    compression_options: list of strings

    :returns: filter_id, compression_opts
    """
    assert check_zfp_availability(), "Attempting to use ZFP filter which is not available."
    # The uniq filter id given by HDF5
    zfp_filter_id = 32013

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

    return zfp_filter_id, compression_opts


def zfp_rate_opts(rate) -> tuple:
    """Create compression options for ZFP in fixed-rate mode

    The float rate parameter is the number of compressed bits per value.
    Parameters:
        :rate: : float
    :rtype: tuple
    """
    zfp_mode_rate = 1
    from struct import pack, unpack
    rate = pack('<d', rate)  # Pack as IEEE 754 double
    high = unpack('<I', rate[0:4])[0]  # Unpack high bits as unsigned int
    low = unpack('<I', rate[4:8])[0]  # Unpack low bits as unsigned int
    return zfp_mode_rate, 0, high, low, 0, 0


def zfp_precision_opts(precision):
    """Create a compression options for ZFP in fixed-precision mode

    The float precision parameter is the number of uncompressed bits per value.

        Parameters:
        :precision: : float
    :rtype: tuple
    """
    zfp_mode_precision = 2
    return zfp_mode_precision, 0, precision, 0, 0, 0


def zfp_accuracy_opts(accuracy):
    """Create compression options for ZFP in fixed-accuracy mode

    The float accuracy parameter is the absolute error tolerance (e.g. 0.001).

        Parameters:
        :accuracy: : float
        :rtype: tuple
    """
    zfp_mode_accuracy = 3
    from struct import pack, unpack
    accuracy = pack('<d', accuracy)  # Pack as IEEE 754 double
    high = unpack('<I', accuracy[0:4])[0]  # Unpack high bits as unsigned int
    low = unpack('<I', accuracy[4:8])[0]  # Unpack low bits as unsigned int
    return zfp_mode_accuracy, 0, high, low, 0, 0


def zfp_expert_opts(minbits, maxbits, maxprec, minexp) -> tuple:
    """Create compression options for ZFP in "expert" mode

    See the ZFP docs for the meaning of the parameters.
    """
    zfp_mode_expert = 4
    return zfp_mode_expert, 0, minbits, maxbits, maxprec, minexp


def zfp_reversible() -> tuple:
    """Create compression options for ZFP in reversible mode

    It should result in lossless compression.
    :rtype: tuple
    """
    zfp_mode_reversible = 5
    return zfp_mode_reversible, 0, 0, 0, 0, 0


