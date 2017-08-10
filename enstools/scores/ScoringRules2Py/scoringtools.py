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
    """Sample Energy Score;
    Compute the energy score ES(*y*, *dat*), where *y* is a vector of a
    *d*-dimensional observation and dat is a multivariate ensemble
    forecast.
    For details, see [1]_.

    Parameters
    ----------
    y : np.array
            Realized values (numeric vector of length *d*)
    dat : np.array
            Forecast sample of shape (*d*, *m*), where
            *d* is the dimension of the realization and
            *m* the number of sample members. Each of the *m* columns corresponds
            to the *d*-dimensional forecast of one ensemble member.

    Returns
    -------
    float
            Energy score of the forecast-observation pair.

    References
    ----------
    .. [1] Gneiting, T., Stanberry, L.I., Grimit, E.P.,
       Held, L. and Johnson, N.A. (2008). Assessing probabilistic forecasts of
       multivariate quantities, with an application to ensemble predictions of
       surface winds. Test, 17, 211–235.
    """
    return __r_caller("es_sample", y, dat)


es_sample_vec = enstools.core.vectorize_multivariate_two_arg(es_sample, arrays_concatenated=False)
es_sample_vec_cat = enstools.core.vectorize_multivariate_two_arg(es_sample)
es_sample_vec_cat.__doc__ = """Sample Energy Score; vectorized version
    Compute the energy score ES(*y_arr*, *dat_arr*), where *y_arr* is a series of 
    *d*-dimensional observations and *dat_arr* is a series of 
    samples of multivariate forecasts.
    For details, see [1]_.
    
    Parameters
    ----------
    y_arr : np.array
            Series of observations of 
            shape (*d*, *n*), where *d* is the dimension of the observations,
            and *n* the number of observation. Hence each column contains a single 
            *d*-dimensional realization.
        
    dat_arr : np.array
            Forecast sample  
            of shape (*d*, *m*, *n*), where
            *d* is the dimension of the realized values, *m* the number of 
            samples, and *n* the number of realizations.
    
    Returns
    -------
    np.array
            Energy score of each forecast-observation pair.

    References
    ----------
    .. [1] Gneiting, T., Stanberry, L.I., Grimit, E.P.,
       Held, L. and Johnson, N.A. (2008). Assessing probabilistic forecasts of
       multivariate quantities, with an application to ensemble predictions of
       surface winds. Test, 17, 211–235.
    """


def vs_sample(y, dat, w=None, p=0.5):
    """Sample Variogram Score;
    Compute the variogram score VS(*y*, *dat*) of order *p*, where *y* is a
    *d*-dimensional observation and dat is a multivariate ensemble
    forecast.
    For details, see [1]_.

    Parameters
    ----------
    y : np.array
            Observation (numeric vector of length *d*).

    dat : np.array
            Forecast sample of shape (*d*, *m*), where
            *d* is the dimension of the realization and
            *m* the number of sample members.

    p : float
            Order of variogram score. Standard choices include *p* = 1 and
            *p* = 0.5 (default).

    w : np.array
            Numeric array of weights for *dat* used in the variogram
            score.  If no weights are specified, constant weights with *w*
            = 1 are used.

    Returns
    -------
    float
            Variogram score of the forecast-observation pair.

    References
    ----------
    .. [1] Scheuerer, M. and Hamill, T.M. (2015). Variogram-based
       proper scoring rules for probabilistic forecasts of multivariate quantities.
       Monthly Weather Review, 143, 1321-1334.
    """
    return __r_caller("vs_sample", y, dat, w, p)


vs_sample_vec = enstools.core.vectorize_multivariate_two_arg(vs_sample, arrays_concatenated=False)
vs_sample_vec_cat = enstools.core.vectorize_multivariate_two_arg(vs_sample)
vs_sample_vec_cat.__doc__ = """
    vs_sample_vec_cat(y_arr, dat_arr, p=0.5, w=1)
    Sample Variogram Score; vectorized version
    Compute the variogram score VS(*y_arr*, *dat_arr*), where *y_arr* is a series of 
    *d*-dimensional observations and *dat_arr* is a series of 
    samples of multivariate forecasts.
    For details, see [1]_.
    
    Parameters
    ----------
    y_arr : np.array
            Series of observations of 
            shape (*d*, *n*), where *d* is the dimension of the observations,
            and *n* the number of observation. Hence each column contains a single 
            *d*-dimensional realization.
        
    dat_arr : np.array
            Forecast sample  
            of shape (*d*, *m*, *n*), where
            *d* is the dimension of the realized values, *m* the number of 
            samples, and *n* the number of realizations.
        
    p : float
            Order of variogram score. Standard choices include *p* = 1 and
            *p* = 0.5 (default).
        
    w : np.array
            Numeric array of weights for *dat* used in the variogram
            score.  If no weights are specified, constant weights with *w*
            = 1 are used.
    
    Returns
    -------
    np.array
            Variogram score of each forecast-observation pair.

    References
    ----------
    .. [1] Scheuerer, M. and Hamill, T.M. (2015). Variogram-based
       proper scoring rules for probabilistic forecasts of multivariate quantities.
       Monthly Weather Review, 143, 1321-1334.
    """


def crps_sample(y, dat, method="edf"):
    """Sample Continuous Ranked Probability Score (CRPS);
    Compute CRPS(*y*, *dat*), where *y* is a univariate
    observation and *dat* is an ensemble forecasts.
    For details, see [1]_.

    Parameters
    ----------
    y : float
            Observation.

    dat : np.array
            Forecast ensemble of length *m*, where
            *m* is the number of members.

    Returns
    -------
    float
            CRPS of the forecast-observation pair.

    References
    ----------
    .. [1] Matheson, J.E. and Winkler, R.L. (1976). Scoring rules for
       continuous probability distributions. Management Science, 22, 1087-1096.
    """
    return __r_caller("crps_sample", y, dat, method)


crps_sample_vec = enstools.core.vectorize_univariate_two_arg(crps_sample)
crps_sample_vec.__doc__ = """
    crps_sample_vec(y_arr, dat_arr, mean=False)
    Sample Continuous Ranked Probability Score (CRPS); vectorized version
    Compute CRPS(*y_arr*, *dat_arr*), where *y_arr* is a series of 
    univariate observations and *dat_arr* is a series of
    ensemble forecasts.
    For details, see [1]_.
    
    Parameters
    ----------
    y_arr : np.array
            Series of observations of 
            length *n*, where *n* is the number of observations. 
        
    dat_arr : np.array
            Ensemble forecasts  
            of shape (*m*, *n*), where *m* is the number of ensemble members, 
            and *n* the number of observation.
            
    mean : bool
            if True, the mean value of the CRPS calculated for each grid point es returned. 
            Otherwise an array with the same dimension as *y_arr* is returned.
    
    Returns
    -------
    np.array or float
            CRPS of each forecast-observation pair.

    References
    ----------
    .. [1] Matheson, J.E. and Winkler, R.L. (1976). Scoring rules for
       continuous probability distributions. Management Science, 22, 1087-1096.
    """
