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
from multipledispatch import dispatch

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
    result = getattr(srl, args[0])(*r_args)
    if len(result) == 1:
        result = result[0]
    else:
        result = rpy2.robjects.numpy2ri.ri2py(result)

    # convert back to xarray when input is an xarray
    if input_is_xarray and type(result) == np.ndarray:
        result = xarray.DataArray(result)

    return result


@dispatch((np.ndarray, xarray.DataArray), (np.ndarray, xarray.DataArray))
def es_sample(obs, fct, mean=False):
    """Sample Energy Score

    Compute the energy score ES(*obs*, *fct*), where *obs* is a vector of a
    *d*-dimensional observation and *fct* is a multivariate ensemble forecast.

    For details, see [1]_.

    Parameters
    ----------
    obs : np.ndarray
            Series of observations of
            shape (*d*, *n*), where *d* is the dimension of the observations,
            and *n* the number of observation. Hence each column contains a single
            *d*-dimensional realization. The dimension *n* may be omitted for single
            observations.

    fct : np.ndarray
            Forecast sample
            of shape (*d*, *m*, *n*), where
            *d* is the dimension of the realized values, *m* the number of
            samples, and *n* the number of realizations. The dimension *n* may be omitted
            for single observations.

    mean : bool
            Applicable for multiple observations only. Is True, the mean value of all scores
            is returned. Otherwise a grid with the same dimension as the *obs* array is returned.

    Returns
    -------
    float or np.ndarray
            Energy score of the forecast-observation pair.

    References
    ----------
    .. [1] Gneiting, T., Stanberry, L.I., Grimit, E.P.,
       Held, L. and Johnson, N.A. (2008). Assessing probabilistic forecasts of
       multivariate quantities, with an application to ensemble predictions of
       surface winds. Test, 17, 211–235.
    """
    if obs.ndim == 1:
        return __r_caller("es_sample", obs, fct)
    if obs.ndim >= 2:
        return __es_sample_vec_cat(obs, fct, mean)


@dispatch((np.ndarray, xarray.DataArray), (np.ndarray, xarray.DataArray), (np.ndarray, xarray.DataArray), (np.ndarray, xarray.DataArray))
def es_sample(obs0, obs1, fct0, fct1, mean=False):
    """Sample Energy Score

    Same computation as in es_sample, but for the special case of two observation and forecast variables.

    Parameters
    ----------
    obs0 : np.ndarray
            Series of observations for the first variable of
            shape (*n*), where *n* is the number of observation.

    obs1 : np.ndarray
            Series of observations for the second variable of
            shape (*n*), where *n* is the number of observation.

    fct0 : np.ndarray
            Forecast sample for the first variable of shape (*m*, *n*), where *m* is the number of
            samples, and *n* the number of realizations.

    fct1 : np.ndarray
            Forecast sample for the first variable of shape (*m*, *n*), where *m* is the number of
            samples, and *n* the number of realizations.

    mean : bool
            Applicable for multiple observations only. Is True, the mean value of all scores
            is returned. Otherwise a grid with the same dimension as the *obs* array is returned.

    Returns
    -------
    float or np.ndarray
            Energy score of the forecast-observation pair.

    References
    ----------
    .. [1] Gneiting, T., Stanberry, L.I., Grimit, E.P.,
       Held, L. and Johnson, N.A. (2008). Assessing probabilistic forecasts of
       multivariate quantities, with an application to ensemble predictions of
       surface winds. Test, 17, 211–235.
    """
    return __es_sample_vec(obs0, obs1, fct0, fct1, mean)


__es_sample_vec = enstools.core.vectorize_multivariate_two_arg(es_sample.dispatch(np.ndarray, np.ndarray), arrays_concatenated=False)
__es_sample_vec_cat = enstools.core.vectorize_multivariate_two_arg(es_sample.dispatch(np.ndarray, np.ndarray))


@dispatch((np.ndarray, xarray.DataArray), (np.ndarray, xarray.DataArray))
def vs_sample(obs, fct, w=None, p=0.5, mean=False):
    """Sample Variogram Score

    Compute the variogram score VS(*obs*, *fct*) of order *p*, where *obs* is a
    *d*-dimensional observation and dat is a multivariate ensemble
    forecast.

    For details, see [1]_.

    Parameters
    ----------
    obs : np.ndarray
            Series of observations of
            shape (*d*, *n*), where *d* is the dimension of the observations,
            and *n* the number of observation. Hence each column contains a single
            *d*-dimensional realization. The dimension *n* may be omitted for single
            observations.

    fct : np.ndarray
            Forecast sample
            of shape (*d*, *m*, *n*), where
            *d* is the dimension of the realized values, *m* the number of
            samples, and *n* the number of realizations. The dimension *n* may be omitted
            for single observations.

    p : float
            Order of variogram score. Standard choices include *p* = 1 and
            *p* = 0.5 (default).

    w : np.ndarray
            Numeric array of weights for *dat* used in the variogram
            score.  If no weights are specified, constant weights with *w*
            = 1 are used.

    mean : bool
            Applicable for multiple observations only. Is True, the mean value of all scores
            is returned. Otherwise a grid with the same dimension as the *obs* array is returned.

    Returns
    -------
    float or np.ndarray
            Variogram score of the forecast-observation pair.

    References
    ----------
    .. [1] Scheuerer, M. and Hamill, T.M. (2015). Variogram-based
       proper scoring rules for probabilistic forecasts of multivariate quantities.
       Monthly Weather Review, 143, 1321-1334.
    """
    if obs.ndim == 1:
        return __r_caller("vs_sample", obs, fct, w, p)
    if obs.ndim >= 2:
        return __vs_sample_vec_cat(obs, fct, mean, w=w, p=p)


@dispatch((np.ndarray, xarray.DataArray), (np.ndarray, xarray.DataArray), (np.ndarray, xarray.DataArray), (np.ndarray, xarray.DataArray))
def vs_sample(obs0, obs1, fct0, fct1, w=None, p=0.5, mean=False):
    """Sample Variogram Score;
    Compute the variogram score VS(*y*, *dat*) of order *p*, where *y* is a
    *d*-dimensional observation and dat is a multivariate ensemble
    forecast.
    For details, see [1]_.

    Parameters
    ----------
    obs0 : np.ndarray
            Series of observations for the first variable of
            shape (*n*), where *n* is the number of observation.

    obs1 : np.ndarray
            Series of observations for the second variable of
            shape (*n*), where *n* is the number of observation.

    fct0 : np.ndarray
            Forecast sample for the first variable of shape (*m*, *n*), where *m* is the number of
            samples, and *n* the number of realizations.

    fct1 : np.ndarray
            Forecast sample for the first variable of shape (*m*, *n*), where *m* is the number of
            samples, and *n* the number of realizations.

    p : float
            Order of variogram score. Standard choices include *p* = 1 and
            *p* = 0.5 (default).

    w : np.ndarray
            Numeric array of weights for *dat* used in the variogram
            score.  If no weights are specified, constant weights with *w*
            = 1 are used.

    mean : bool
            Applicable for multiple observations only. Is True, the mean value of all scores
            is returned. Otherwise a grid with the same dimension as the *obs* array is returned.

    Returns
    -------
    float or np.ndarray
            Variogram score of the forecast-observation pair.

    References
    ----------
    .. [1] Scheuerer, M. and Hamill, T.M. (2015). Variogram-based
       proper scoring rules for probabilistic forecasts of multivariate quantities.
       Monthly Weather Review, 143, 1321-1334.
    """
    return __vs_sample_vec(obs0, obs1, fct0, fct1, mean, w=w, p=p)


__vs_sample_vec = enstools.core.vectorize_multivariate_two_arg(vs_sample.dispatch(np.ndarray, np.ndarray), arrays_concatenated=False)
__vs_sample_vec_cat = enstools.core.vectorize_multivariate_two_arg(vs_sample.dispatch(np.ndarray, np.ndarray))


def crps_sample(y, dat, mean=False):
    """Sample Continuous Ranked Probability Score (CRPS);
    Compute CRPS(*y*, *dat*), where *y* is a univariate
    observation and *dat* is an ensemble forecasts.
    For details, see [1]_.

    Parameters
    ----------
    y : float or np.ndarray or xarray.DataArray
            Single observation or *n*-dimensional array of observations. The observations are allowed to contain missing
            values in form of NaNs.

    dat : np.ndarray or xarray.DataArray
            Ensemble forecasts *m* for a single observation or of shape (*m*, *n*),
            where *m* is the number of ensemble members,
            and *n* the number of observation.

    mean : bool
            If True, the mean value of all calculated CRPS values is returned. Otherwise
            an array with the same shape as the observations is returned.

    Returns
    -------
    float or np.ndarray
            CRPS of the forecast-observation pair.

    References
    ----------
    .. [1] Matheson, J.E. and Winkler, R.L. (1976). Scoring rules for
       continuous probability distributions. Management Science, 22, 1087-1096.
    """
    if not hasattr(y, "shape") or len(y.shape) == 0:
        # is the observation is nan, return nan as result
        if np.isnan(y):
            return np.nan
        else:
            return __r_caller("crps_sample", y, dat)
    else:
        # are there any nans? if os, replace them by zeros and remember their position
        nan_index = np.where(np.isnan(y))
        if len(nan_index) > 0:
            y = y.copy()
            dat = dat.copy()
            y[nan_index] = 0
            dat[:, nan_index] = 0
        if len(y.shape) > 1:
            # The R-function expects one dimensional forecasts and 1d observations
            original_shape = y.shape
            y_flat = y.flatten()
            dat_reshaped = dat.reshape((dat.size // y_flat.size, y_flat.size))
            result = __r_caller("crps_sample", y_flat, np.moveaxis(dat_reshaped, 0, -1))
            result = result.reshape(original_shape)
        else:
            result = __r_caller("crps_sample", y, np.moveaxis(dat, 0, -1))
        # put back the nans into the array, if any
        if len(nan_index) > 0:
            result[nan_index] = np.nan
        if mean:
            result = np.nanmean(result)
        return result
