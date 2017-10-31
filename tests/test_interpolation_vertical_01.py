import enstools.interpolation.vertical_interpolation
import numpy as np


def test_get_weights():
    """
    test for enstools.interpolation.vertical_interpolation.get_weights
    """
    src_p = np.empty((10, 2))
    src_p[:,0] = np.arange(100, 1100, 100)
    src_p[:,1] = src_p[:,0] + 10
    dst_p = np.asarray([250])

    # callculate indices and weights for these indices
    indices, weights = enstools.interpolation.vertical_interpolation.get_weights(src_p, dst_p)

    # the destination is in between level 1 and 2 for both cells
    np.testing.assert_array_equal(indices[:, :, 0], 1)
    np.testing.assert_array_equal(indices[:, :, 1], 2)

    # in the first cell, the value is in between both levels, in the second, the value in closer to the first
    np.testing.assert_almost_equal(weights[0, 0, :], 0.5)
    np.testing.assert_almost_equal(weights[0, 1, :], [0.6, 0.4])

    # ---------------------------------------------------------------
    # next test. one destination pressure is excatly one source level
    dst_p = np.asarray([300])
    indices, weights = enstools.interpolation.vertical_interpolation.get_weights(src_p, dst_p)

    # the destination on the index 2 for the frist cell and between 1 and 2 for the second cell
    np.testing.assert_array_equal(indices[0, 0, :], [2, 3])
    np.testing.assert_array_equal(indices[0, 1, :], [1, 2])

    # in the first cell, the value is in between both levels, in the second, the value in closer to the first
    np.testing.assert_almost_equal(weights[0, 0, :], [1.0, 0.0])
    np.testing.assert_almost_equal(weights[0, 1, :], [0.1, 0.9])

    # ---------------------------------------------------------------
    # next test. one destination is above all source values, one if below all source values
    dst_p = np.asarray([0, 1100])
    indices, weights = enstools.interpolation.vertical_interpolation.get_weights(src_p, dst_p)

    # all weights are zero
    np.testing.assert_almost_equal(weights, 0.0)


def test_model2pressure():
    """
    test of the model2pressure interpolator
    """
    src_p = np.empty((10, 2))
    src_p[:,0] = np.arange(100, 1100, 100)
    src_p[:,1] = src_p[:,0] + 10
    dst_p = np.asarray([250, 350])

    # create the interpolator object
    intp = enstools.interpolation.model2pressure(src_p, dst_p)

    # interpolate the pressure field onto pressure levels, that should result in exactly the destination values
    new_p = intp(src_p)
    np.testing.assert_almost_equal(new_p, [[250, 250], [350, 350]])

    # create a source field with lon and lat
    src_p = np.empty((10, 4, 6))
    for p in range(100, 1100, 100):
        src_p[p//100-1, ...] = p
    # add some random deviations
    src_p += np.random.randn(*src_p.shape) * 10

    # create interpolator object
    intp = enstools.interpolation.model2pressure(src_p, [500, 850])
    new_p = intp(src_p)
    np.testing.assert_almost_equal(new_p[0, ...], np.ones((4, 6)) * 500)
    np.testing.assert_almost_equal(new_p[1, ...], np.ones((4, 6)) * 850)
    np.testing.assert_array_equal(new_p.shape, (2, 4, 6))
    print(new_p)

    intp = enstools.interpolation.model2pressure(src_p, 500)
