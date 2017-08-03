import xarray
import numpy
import enstools


def test_embed_image():
    """
    test of embed_image from match_pyramide_ic
    """
    # create test image
    test_im = xarray.DataArray(numpy.random.randn(5, 3))

    # new array should have shape (8, 4)
    result = enstools.scores.DisplacementAmplitudeScore.match_pyramid_ic.embed_image(test_im, 4)
    numpy.testing.assert_array_equal(numpy.array(result.shape), numpy.array((8, 4)))

    # new array should have shape (24, 6)
    result = enstools.scores.DisplacementAmplitudeScore.match_pyramid_ic.embed_image(test_im, 4, 3, 3)
    numpy.testing.assert_array_equal(numpy.array(result.shape), numpy.array((24, 6)))

    # input image should be part of result image
    numpy.testing.assert_array_equal(test_im, result[:5, :3])


def test_map_backwards():
    """
    test of backward mapping from match_pyramide_ic
    """
    # create test image
    test_im = numpy.zeros((5, 5))
    test_im[2, 2] = 1

    # create displacement vectors
    xdis = numpy.ones((5, 5))
    ydis = xdis

    # apply mapping
    result = enstools.scores.DisplacementAmplitudeScore.match_pyramid_ic.map_backward(test_im, xdis, ydis)
    expected = numpy.zeros((5, 5))
    expected[1, 1] = 1
    numpy.testing.assert_array_equal(result, expected)


def test_gauss_kern():
    """
    test of gauss_kern from match_pyramide_ic
    """
    result = enstools.scores.DisplacementAmplitudeScore.match_pyramid_ic.gauss_kern(1, 1)
    numpy.testing.assert_equal(result.sum(), 1)


def test_downsize():
    """
    test of downsize from match_pyramid
    """
    # create test image
    test_image = numpy.random.randn(4, 4)

    # downsize by factor 2
    result = enstools.scores.DisplacementAmplitudeScore.match_pyramid_ic.downsize(test_image, 2)
    numpy.testing.assert_equal(result[0, 0], test_image[0:2, 0:2].mean())


def test_match_pyramid():
    """
    test of match_pyramid from match_pyramid
    """
    # create two test images
    im1 = numpy.zeros((5, 5))
    im1[1:3, 1:3] = 1
    im2 = numpy.zeros((5, 5))
    im2[2:4, 2:4] = 1

    result, xdis, ydis, lse = enstools.scores.DisplacementAmplitudeScore.match_pyramid_ic.match_pyramid(im1, im2)
    numpy.testing.assert_array_almost_equal(numpy.round(result), im2)


def test_calc_das():
    """
    test of pure das calculation calc_das from calc_das.py
    """
    # create two test images
    obs = numpy.zeros((5, 5))
    obs[1:3, 1:3] = 1
    fct = numpy.zeros((5, 5))
    fct[2:4, 2:4] = 1

    # morph fct to obs,obs-space
    morph_o, xdis_o, ydis_o, lse_o = enstools.scores.DisplacementAmplitudeScore.match_pyramid_ic.match_pyramid(fct, obs)
    # morph obs to fct,fct-space
    morph_f, xdis_f, ydis_f, lse_f = enstools.scores.DisplacementAmplitudeScore.match_pyramid_ic.match_pyramid(obs, fct)

    # reproduce expected values
    das, dis, amp, rms_obs = enstools.scores.DisplacementAmplitudeScore.calc_das.calc_das(obs, fct, xdis_o, ydis_o,
                                                                                          lse_o, xdis_f, ydis_f, lse_f,
                                                                                          dis_max=5, threshold=0.5)
    expected = (0.48602544875444409, 0.35238775926722798, 0.1336376894872161, 1.0)
    numpy.testing.assert_array_almost_equal((das, dis, amp, rms_obs), expected)


def test_threshold_data():
    """
    test of threshold data from calc_das
    """
    # create test data
    obs = numpy.random.randn(10, 10)
    sum_obs = numpy.sum(obs)

    # set everything below 1 to zero
    filtered = enstools.scores.DisplacementAmplitudeScore.calc_das.threshold_data(obs, 1)
    for x in range(10):
        for y in range(10):
            numpy.testing.assert_equal(filtered[x, y] == 0 or filtered[x, y] > 1, True)

    # the input array should remain unchanged
    numpy.testing.assert_equal(numpy.sum(obs), sum_obs)


def test_das():
    """
    test of the actual DAS score
    """
    # create test data
    obs = numpy.zeros((100, 100))
    obs[50:52, 50:52] = 2
    fct = numpy.zeros((100, 100))
    fct[51:53, 51:53] = 2

    # perform calculation
    das = enstools.scores.DisplacementAmplitudeScore.das(obs, fct)
    numpy.testing.assert_array_almost_equal(das["das"], 0.857092469745)
    numpy.testing.assert_array_almost_equal(das["dis"], 0.027265825324)
    numpy.testing.assert_array_almost_equal(das["amp"], 0.829826644421)
    numpy.testing.assert_array_almost_equal(das["rms_obs"], 0.11111111)

    # perfect score
    das = enstools.scores.DisplacementAmplitudeScore.das(obs, obs)
    numpy.testing.assert_array_almost_equal(das["das"], 0.0)
    numpy.testing.assert_array_almost_equal(das["dis"], 0.0)
    numpy.testing.assert_array_almost_equal(das["amp"], 0.0)

    # only values below threshold
    obs[50:52, 50:52] = 1
    fct[51:53, 51:53] = 1
    das = enstools.scores.das(obs, fct, threshold=1)
    numpy.testing.assert_array_equal(das["das"], numpy.nan)
    numpy.testing.assert_array_equal(das["dis"], numpy.nan)
    numpy.testing.assert_array_equal(das["amp"], numpy.nan)
    numpy.testing.assert_array_equal(das["rms_obs"], numpy.nan)
