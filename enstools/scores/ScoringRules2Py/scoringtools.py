# -*- coding: utf-8 -*-
"""
This module is an interface for some functionality of the R-package scoringRules.
It is possible to calculate
various scores for ensemble predictions. Namely the Continuous Ranked 
Probability Score (CRPS), the energy score and
the variogram score.
"""

import numpy as np
import importlib
import xarray
import enstools

rpy2 = None
srl = None


def __init_R():
    """
    Initialize the embedded R process when this module is used for the first time. That is necessary to make the R
    setup optional. enstools would not be usable otherwise.
    """
    global srl
    global rpy2
    if srl is None:
        rpy2 = importlib.import_module("rpy2")
        import rpy2.robjects.packages
        import rpy2.robjects.numpy2ri
        srl = rpy2.robjects.packages.importr('scoringRules')


# list of argument types for R functions that don't need any conversion
directly_supported_r_args = [float, np.float32, np.float64, str]

# list of data types for which an conversion to float is required
conversion_to_float_r_args = [int, np.int32, np.int64]


def __r_caller(*args):
    """
    Call a R function and translate the input and output

    Parameters
    ----------
    args : int, float, double, numpy.ndarray, xarray.DataArray
            Arguments for the R-function. The first argument has to be the name of the R-function

    Returns
    -------
    return value of the R-function
    """
    # start the R-process if not yet running
    __init_R()

    # translate arguments
    r_args = []
    input_is_xarray = False
    for arg in args[1:]:
        # float and strings are supported directly
        if type(arg) in directly_supported_r_args:
            r_args.append(arg)
        # int is converted to float
        elif type(arg) in conversion_to_float_r_args:
            r_args.append(float(arg))
        # numpy arrays are converted
        elif type(arg) == np.ndarray:
            if arg.ndim == 1:
                r_args.append(rpy2.robjects.FloatVector(arg))
            else:
                r_args.append(rpy2.robjects.numpy2ri.py2ri(arg))
        # for xarrays the underlying numpy array is used
        elif type(arg) == xarray.DataArray:
            input_is_xarray = True
            if arg.ndim == 1:
                r_args.append(rpy2.robjects.FloatVector(arg.data))
            else:
                r_args.append(rpy2.robjects.numpy2ri.py2ri(arg.data))
        elif arg is None:
            r_args.append(rpy2.rinterface.NULL)
        # any other arguments are not supported
        else:
            raise ValueError("Unsupported argument type: %s" % type(arg))

    # call the R-function
    result = getattr(srl, args[0])(*r_args)[0]

    # convert back to xarray when input is an xarray
    if input_is_xarray and type(result) == np.ndarray:
        result = xarray.DataArray(result)

    return result


def es_sample(y, dat):
    """Sample Energy Score

    This function calculates the energy score for a given pair of
    observation (y) and ensemble prediction (dat). The observation
    y is considered to be multivariate.

    Args:
        y (np.array): Multivariate observation of length d greater 
                            than 1.
        dat (np.array): Ensemble prediction for value y containing 
        d times m values, where m is the 
        number of ensemble members.

    Returns:
        float: Returns the energy score of the forecast-observation pair.

    """
    return __r_caller("es_sample", y, dat)


es_sample_vec_cat = enstools.core.vectorize_multivariate_two_arg(es_sample)
es_sample_vec = enstools.core.vectorize_multivariate_two_arg(es_sample, arrays_concatenated=False)


def vs_sample(y, dat, w=None, p=0.5):
    """Sample Variogram Score

    This function calculates the variogram score for a given pair of
    observation (y) and ensemble prediction (dat). The observation
    y is considered to be multivariate.

    Args:
        y (np.array): Multivariate observation of length d greater 
                            than 1.
        dat (np.array): Ensemble prediction for value y containing 
        d times m values, where m is the 
        number of ensemble members.

    Returns:
        float: Returns the variogram score of the forecast-observation pair.

    """
    return __r_caller("vs_sample", y, dat, w, p)


vs_sample_vec_cat = enstools.core.vectorize_multivariate_two_arg(vs_sample)
vs_sample_vec = enstools.core.vectorize_multivariate_two_arg(vs_sample, arrays_concatenated=False)


def crps_sample(y, dat, method="edf"):
    """Sample Continuous Ranked Probability Score (CRPS)

    This function calculates the CRPS for a given pair of
    observation (y) and ensemble prediction (dat). The observation
    y is considered to be univariate.

    Args:
        y (float): Univariate observation.
        
        dat (np.array): Ensemble prediction for value y of length m which is
        the number of ensemble members.

    Returns:
        float: Returns the CRPS of the forecast-observation pair.

    """
    return __r_caller("crps_sample", y, dat, method)


crps_sample_vec = enstools.core.vectorize_univariate_two_arg(crps_sample)
