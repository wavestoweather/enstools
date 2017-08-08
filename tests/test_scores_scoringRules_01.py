import xarray
import numpy
import enstools
import timeit
import numpy as np
import rpy2
import rpy2.robjects.numpy2ri as np2ri
from rpy2.robjects.packages import importr
srl = importr('scoringRules')


def test_crps_sample():
    """
    test of crps_sample from scoringtools.py
    """
    # create example data
    x = numpy.random.randn(100000)
    y = xarray.DataArray(x)

    # first argument int, second numpy
    res = enstools.scores.crps_sample(1, x)
    numpy.testing.assert_almost_equal(res, 0.6, decimal=2)

    # first argument float, second numpy
    res = enstools.scores.crps_sample(1.0, x)
    numpy.testing.assert_almost_equal(res, 0.6, decimal=2)

    # first argument float, second xarray
    res = enstools.scores.crps_sample(1.0, y)
    numpy.testing.assert_almost_equal(res, 0.6, decimal=2)


def test_crps_sample_vec():
    """
    test of crps_sample vectorization
    """
    # create example data
    obs = numpy.ones(100000)
    fct = numpy.random.randn(20, 100000)
    res = enstools.scores.crps_sample_vec(obs, fct)
    res2 = enstools.scores.crps_sample_vec2(obs, fct, mean=True)
    res3 = enstools.scores.crps_sample_vec2(numpy.reshape(obs, (100, 1000)), numpy.reshape(fct, (20, 100, 1000)), mean=True)
    #t = timeit.Timer("enstools.scores.crps_sample_vec2(obs, fct)", setup="""gc.enable() ; import enstools ; import numpy ; obs = numpy.ones(194081) ;fct = numpy.random.randn(20, 194081)""")
    #print(t.timeit(10))
    print(res, res2, res3)


def test_es_sample_vec():
    """
    test of es_sample vectorization
    """
    obs = numpy.ones((2, 10000))
    fct = numpy.random.randn(2, 20, 10000)
    res = enstools.scores.es_sample_vec(obs, fct)
    res2 = enstools.scores.es_sample_vec2(obs, fct)
    res3 = enstools.scores.es_sample_vec2(numpy.reshape(obs, (2, 100, 100)), numpy.reshape(fct, (2, 20, 100, 100)))
    res4 = enstools.scores.es_sample_vec3(obs[0,:], obs[1,:], fct[0,:,:], fct[1,:,:])
    #print(res, res2, res2.shape, numpy.mean(res2))
    #print(res, res3, res3.shape, numpy.mean(res3))
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