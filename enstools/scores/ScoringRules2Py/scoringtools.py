# -*- coding: utf-8 -*-
"""
This module is an interface for some functionality of the R-package scoringRules.
It is possible to calculate
various scores for ensemble predictions. Namely the Continuous Ranked 
Probability Score (CRPS), the energy score and
the variogram score.
"""

import numpy as np
import rpy2
import rpy2.robjects.numpy2ri as np2ri
from rpy2.robjects.packages import importr
srl = importr('scoringRules')

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
    try:
        y = np.array(y)
        dat = np.array(dat)
        y_r = rpy2.robjects.FloatVector(y)
        dat_r = np2ri.py2ri(dat)
    except Exception:
        print('Input has wrong format.')
        
    return srl.es_sample(y_r, dat_r)[0]
    
def es_sample_vec(y_mat, dat_mat):
    """Sample Energy Score; vectorized version

    This function calculates the energy score for a given pair of
    a series of observations (y_mat) and 
    corresponding ensemble predictions (dat_mat). The data in
    y_mat is considered to be a series of multivariate observations.

    Args:
        y_mat (np.array): Series of multivariate observations of 
        dimension d times n. Where d is the dimension of the multivariate
        observation and n the number of observations.
        
        dat (np.array): Ensemble prediction for the values y_mat containing
        d times m times n values where m is the number of ensemble members.

    Returns:
        float: Returns the energy score of the forecast-observation series.

    """
    try:
        y_mat = np.array(y_mat)
        y_mat = np.expand_dims(y_mat, 1)
        dat_mat = np.array(dat_mat)
    except Exception:
        print('Input has wrong format.')
    else:
        if (len(y_mat.shape) != 3 
            or len(dat_mat.shape) != 3
            or y_mat.shape[0] != dat_mat.shape[0]
            or y_mat.shape[2] != dat_mat.shape[2]
        ):
            raise ValueError('Parameters have wrong dimension.')

    df = np.concatenate((y_mat,dat_mat),axis = 1)
    df_r = np2ri.py2ri(df)
    rpy2.robjects.globalenv['df'] =  df_r
    
    return rpy2.robjects.r('mean(apply(df, c(3), function(x) es_sample(x[,1], x[,-1])))')[0]
    
def vs_sample(y, dat):
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
    try:
        y = np.array(y)
        dat = np.array(dat)
        y_r = rpy2.robjects.FloatVector(y)
        dat_r = np2ri.py2ri(dat)
    except Exception:
        print('Input has wrong format.')

    return srl.vs_sample(y_r, dat_r)[0]
    
def vs_sample_vec(y_mat, dat_mat):
    """Sample Variogram Score; vectorized version

    This function calculates the Energy score for a given pair of
    a series of observations (y_mat) and 
    corresponding ensemble predictions (dat_mat). The data in
    y_mat is considered to be a series of multivariate observations.

    Args:
        y_mat (np.array): Series of multivariate observations of 
        dimension d times n. Where d is the dimension of the multivariate
        observation and n the number of observations.
        
        dat (np.array): Ensemble prediction for the values y_mat containing
        d times m times n values where m is the number of ensemble members.

    Returns:
        float: Returns the variogram score of the forecast-observation series.

    """
    try:
        y_mat = np.array(y_mat)
        y_mat = np.expand_dims(y_mat, 1)
        dat_mat = np.array(dat_mat)
    except Exception:
        print('Input has wrong format.')
    else:
        if (len(y_mat.shape) != 3
            or len(dat_mat.shape) != 3
            or y_mat.shape[0] != dat_mat.shape[0]
            or y_mat.shape[2] != dat_mat.shape[2]
        ):
            raise ValueError('Parameters have wrong dimension.')

    df = np.concatenate((y_mat,dat_mat),axis = 1)
    df_r = np2ri.py2ri(df)
    rpy2.robjects.globalenv['df'] =  df_r
    
    return rpy2.robjects.r('mean(apply(df, c(3), function(x) vs_sample(x[,1], x[,-1])))')[0]
    
def crps_sample(y, dat):
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
    try:
        y_r = float(y)
        dat = np.array(dat)
        dat_r = rpy2.robjects.FloatVector(dat)
    except Exception:
        print('Input has wrong format.')
        
    return srl.crps_sample(y_r, dat_r)[0]
    
def crps_sample_vec(y_vec, dat_mat):
    """Sample Continuous Ranked Probability Score (CRPS); vectorized version

    This function calculates the CRPS for a given pair of
    a series of observations (y_vec) and 
    corresponding ensemble predictions (dat_mat). The data in
    y_vec is considered to be a series of univariate observations.

    Args:
        y_vec (np.array): Series of multivariate observations of 
        length n which is the number of observations.
        
        dat_mat (np.array): Ensemble prediction for the values y_vec containing
        m times n values where m is the number of ensemble members.

    Returns:
        float: Returns the CRPS of the forecast-observation series.

    """
    try:
        y_vec = np.array(y_vec)
        dat_mat = np.array(dat_mat)
        y_r = rpy2.robjects.FloatVector(y_vec)
        dat_r = np2ri.py2ri(dat_mat)
    except Exception:
        print('Input has wrong format.')
    else:
        if (len(y_vec.shape) != 1 
            or len(dat_mat.shape) != 2 
            or y_vec.shape[0] != dat_mat.shape[1]
        ):
            raise ValueError('Parameters have wrong dimension.')

    rpy2.robjects.globalenv['obs'] =  y_r
    rpy2.robjects.globalenv['forc'] =  dat_r
    
    return rpy2.robjects.r('mean(apply(rbind(obs,forc), 2, function(x) crps_sample(x[1], x[-1])))')[0]
