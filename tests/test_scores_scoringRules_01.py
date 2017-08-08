import xarray
import enstools
import timeit
import numpy as np
import rpy2
import rpy2.robjects.numpy2ri as np2ri
from rpy2.robjects.packages import importr
srl = importr('scoringRules')


def test_es_sample():
    """
    test of es_sample
    """
    # compare with reference version
    obs = np.ones(2)
    fct = np.random.randn(2, 20)
    res = enstools.scores.es_sample(obs, fct)
    res2 = __reference__es_sample(obs, fct)
    np.testing.assert_almost_equal(res, res2)


def test_es_sample_vec():
    """
    test of es_sample vectorization
    """
    # compare with reference version for variant with concatenated input arrays
    obs = np.ones((2, 1000))
    fct = np.random.randn(2, 20, 1000)
    res = enstools.scores.es_sample_vec_cat(obs, fct)
    res2 = __reference__es_sample_vec(obs, fct)
    np.testing.assert_array_almost_equal(res, res2)

    # compare to variant with separate input variables
    obs1 = obs[0, :]
    obs2 = obs[1, :]
    fct1 = fct[0, :, :]
    fct2 = fct[1, :, :]
    res3 = enstools.scores.es_sample_vec(obs1, obs2, fct1, fct2)
    np.testing.assert_array_almost_equal(res3, res2)


def test_vs_sample():
    """
    test of vs_sample
    """
    # compare with reference version
    obs = np.ones(2)
    fct = np.random.randn(2, 20)
    res = enstools.scores.vs_sample(obs, fct)
    res2 = __reference__vs_sample(obs, fct)
    np.testing.assert_almost_equal(res, res2)


def test_vs_sample_vec():
    """
    test of vs_sample vectorization
    """
    # compare with reference version for variant with concatenated input arrays
    obs = np.ones((2, 1000))
    fct = np.random.randn(2, 20, 1000)
    res = enstools.scores.vs_sample_vec_cat(obs, fct)
    res2 = __reference__vs_sample_vec(obs, fct)
    np.testing.assert_array_almost_equal(res, res2)

    # compare to variant with separate input variables
    obs1 = obs[0, :]
    obs2 = obs[1, :]
    fct1 = fct[0, :, :]
    fct2 = fct[1, :, :]
    res3 = enstools.scores.vs_sample_vec(obs1, obs2, fct1, fct2)
    np.testing.assert_array_almost_equal(res3, res2)


def test_crps_sample():
    """
    test of crps_sample from scoringtools.py
    """
    # create example data
    x = np.random.randn(1000)
    y = xarray.DataArray(x)

    # first argument int, second numpy
    res = enstools.scores.crps_sample(1, x)
    np.testing.assert_almost_equal(res, 0.6, decimal=2)

    # first argument float, second numpy
    res = enstools.scores.crps_sample(1.0, x)
    np.testing.assert_almost_equal(res, 0.6, decimal=2)

    # first argument float, second xarray
    res = enstools.scores.crps_sample(1.0, y)
    np.testing.assert_almost_equal(res, 0.6, decimal=2)

    # result is equal to reference implementation
    res2 = __reference__crps_sample(1.0, y)
    np.testing.assert_almost_equal(res, res2)


def test_crps_sample_vec():
    """
    test of crps_sample vectorization
    """
    # create example data
    obs = np.ones(1000)
    fct = np.random.randn(20, 1000)

    # test for not averaged result
    res = enstools.scores.crps_sample_vec(obs, fct)
    res2 = __reference__crps_sample_vec(obs, fct)
    np.testing.assert_array_almost_equal(res, res2)

    # test for averaged result
    res = enstools.scores.crps_sample_vec(obs, fct, mean=True)
    res2 = np.mean(__reference__crps_sample_vec(obs, fct))
    np.testing.assert_almost_equal(res, res2)

    # test for gridded data
    obs = obs.reshape(100, 10)
    fct = fct.reshape(20, 100, 10)
    res = enstools.scores.crps_sample_vec(obs, fct, mean=True)
    np.testing.assert_almost_equal(res, res2)


# reference implementation from Sebastian Lerch and Manuel Klar
def __reference__es_sample(y, dat):
    try:
        y = np.array(y)
        dat = np.array(dat)
        y_r = rpy2.robjects.FloatVector(y)
        dat_r = np2ri.py2ri(dat)
    except Exception:
        print('Input has wrong format.')
    return srl.es_sample(y_r, dat_r)[0]


def __reference__es_sample_vec(y_arr, dat_arr):
    try:
        y_arr = np.array(y_arr)
        y_arr = np.expand_dims(y_arr, 1)
        dat_arr = np.array(dat_arr)
    except Exception:
        print('Input has wrong format.')
    else:
        if (len(y_arr.shape) != 3
            or len(dat_arr.shape) != 3
            or y_arr.shape[0] != dat_arr.shape[0]
            or y_arr.shape[2] != dat_arr.shape[2]
            ):
            raise ValueError('Parameters have wrong dimension.')

    df = np.concatenate((y_arr, dat_arr), axis=1)
    df_r = np2ri.py2ri(df)
    rpy2.robjects.globalenv['df'] = df_r

    escr_r = rpy2.robjects.r('apply(df, c(3), function(x) es_sample(x[,1], x[,-1]))')
    return np.array(escr_r)


def __reference__vs_sample(y, dat, w=None, p=0.5):
    try:
        y = np.array(y)
        dat = np.array(dat)
        if w is None:
            w_r = rpy2.robjects.NULL
        else:
            w = np.array(w)
            w_r = np2ri.py2ri(w)
        p_r = float(p)
        y_r = rpy2.robjects.FloatVector(y)
        dat_r = np2ri.py2ri(dat)
    except Exception:
        print('Input has wrong format.')

    return srl.vs_sample(y=y_r, dat=dat_r, w=w_r, p=p_r)[0]


def __reference__vs_sample_vec(y_arr, dat_arr, w=None, p=0.5):
    try:
        y_arr = np.array(y_arr)
        y_arr = np.expand_dims(y_arr, 1)
        dat_arr = np.array(dat_arr)
        p_r = float(p)
        if w is None:
            w_r = rpy2.robjects.NULL
        else:
            w = np.array(w)
            w_r = np2ri.py2ri(w)
    except Exception:
        print('Input has wrong format.')
    else:
        if (len(y_arr.shape) != 3
            or len(dat_arr.shape) != 3
            or y_arr.shape[0] != dat_arr.shape[0]
            or y_arr.shape[2] != dat_arr.shape[2]
            ):
            raise ValueError('Parameters have wrong dimension.')

    df = np.concatenate((y_arr, dat_arr), axis=1)
    df_r = np2ri.py2ri(df)
    rpy2.robjects.globalenv['df'] = df_r
    rpy2.robjects.globalenv['p'] = p_r
    rpy2.robjects.globalenv['w'] = w_r

    vscr_r = rpy2.robjects.r('apply(df, c(3), function(x) vs_sample(x[,1], x[,-1], w, p))')
    return np.array(vscr_r)


def __reference__crps_sample(y, dat):
    try:
        y_r = float(y)
        dat = np.array(dat)
        dat_r = rpy2.robjects.FloatVector(dat)
    except Exception:
        print('Input has wrong format.')

    return srl.crps_sample(y_r, dat_r)[0]


def __reference__crps_sample_vec(y_arr, dat_arr):
    try:
        y_arr = np.array(y_arr)
        dat_arr = np.array(dat_arr)
        y_r = rpy2.robjects.FloatVector(y_arr)
        dat_r = np2ri.py2ri(dat_arr)
    except Exception:
        print('Input has wrong format.')
    else:
        if (len(y_arr.shape) != 1
            or len(dat_arr.shape) != 2
            or y_arr.shape[0] != dat_arr.shape[1]
            ):
            raise ValueError('Parameters have wrong dimension.')

    rpy2.robjects.globalenv['obs'] = y_r
    rpy2.robjects.globalenv['forc'] = dat_r

    crps_r = rpy2.robjects.r('apply(rbind(obs,forc), 2, function(x) crps_sample(x[1], x[-1]))')
    return np.array(crps_r)