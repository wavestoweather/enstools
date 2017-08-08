import xarray
import numpy
import enstools
import timeit


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
