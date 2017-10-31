import enstools
import numpy as np


def test_nearest_neighbour_regular_1d():
    """
    test of nearest_neighbour from enstools.interpolation with 1d-coordinate arrays
    """
    # test with regular grid and 1d coords
    grid_lon = np.arange(100)
    grid_lat = np.arange(50)
    data = np.zeros((50, 100))

    # the four nearest values for the first point
    data[20:22, 10:12] = 7

    # the four nearest values for the second point
    data[17:19, 13:15] = 8

    # the actual test
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, (10.2, 13.2), (20.2, 17.2), npoints=4)(data)
    np.testing.assert_array_almost_equal(res, [7, 8])

    # same test, but with 3d-data (e.g., level, lat, lon)
    data2 = np.zeros((10, 50, 100))
    for i in range(10):
        data2[i, :, :] = data + i

    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, (10.2, 13.2), (20.2, 17.2), npoints=4)(data2)
    np.testing.assert_array_almost_equal(res, np.asarray([np.arange(7, 17, 1), np.arange(8, 18, 1)]).transpose())

    # same test with only one neighbour or only one target point
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, (10.2, 13.2), (20.2, 17.2), npoints=1)(data)
    np.testing.assert_array_almost_equal(res, [7, 8])
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, 10.2, 20.2, npoints=1)(data)
    np.testing.assert_array_almost_equal(res, 7)
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, 13.2, 17.2, npoints=1)(data2)
    np.testing.assert_array_almost_equal(res, np.arange(8, 18, 1).reshape(10, 1))


def test_nearest_neighbour_regular_2d():
    """
    test of nearest_neighbour from enstools.interpolation with 2d-coordinate arrays
    """
    # test with regular grid and 2d coords
    grid_lon, grid_lat = np.meshgrid(np.arange(100), np.arange(50), indexing="ij")
    data = np.zeros((100, 50))

    # the four nearest values for the first point
    data[10:12, 20:22] = 7

    # the four nearest values for the second point
    data[13:15, 17:19] = 8

    # the actual test
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, (10.2, 13.2), (20.2, 17.2), npoints=4)(data)
    np.testing.assert_array_almost_equal(res, [7, 8])

    # same test, but with 3d-data (e.g., level, lon, lat)
    data2 = np.zeros((10, 100, 50))
    for i in range(10):
        data2[i, :, :] = data + i

    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, (10.2, 13.2), (20.2, 17.2), npoints=4)(data2)
    np.testing.assert_array_almost_equal(res, np.asarray([np.arange(7, 17, 1), np.arange(8, 18, 1)]).transpose())

    # same test with one neighbour point
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, 10.2, 20.2, npoints=1)(data2)
    np.testing.assert_array_almost_equal(res, np.arange(7, 17, 1).reshape(10, 1))
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, (10.2, 13.2), (20.2, 17.2), npoints=1)(data2)
    np.testing.assert_array_almost_equal(res, np.asarray([np.arange(7, 17, 1), np.arange(8, 18, 1)]).transpose())


def test_nearest_neighbour_unstructured():
    """
    test of nearest_neighbour from enstools.interpolation with un unstructured grid
    """
    # create coordinates
    grid_lon = np.arange(100)
    grid_lat = np.ones(100)
    data = np.zeros(100)

    # the nearest 3 points
    data[10:13] = 7
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, (11.2, 2.2), (11.2, 13.2), npoints=3, src_grid="unstructured")(data)
    np.testing.assert_array_almost_equal(res, [7, 0])

    # same test, but with 2d-data (e.g., level, ncell)
    data2 = np.zeros((10, 100))
    for i in range(10):
        data2[i, :] = data + i

    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, (11.2, 2.2), (11.2, 13.2), npoints=3, src_grid="unstructured")(data2)
    np.testing.assert_array_almost_equal(res, np.asarray([np.arange(7, 17, 1), np.arange(0, 10, 1)]).transpose())

    # only one point
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, 11.2, 13.2, npoints=3, src_grid="unstructured")(data)
    np.testing.assert_almost_equal(res, 7)
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, 11.2, 13.2, npoints=3, src_grid="unstructured")(data2)
    np.testing.assert_array_almost_equal(res, np.arange(7, 17, 1).reshape(10, 1))

    # same test with one one neighbour point
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, 11.2, 13.2, npoints=1, src_grid="unstructured")(data)
    np.testing.assert_almost_equal(res, 7)
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, 11.2, 13.2, npoints=1, src_grid="unstructured")(data2)
    np.testing.assert_almost_equal(res, np.arange(7, 17, 1).reshape(10, 1))


def test_test_nearest_neighbour_dmean():
    """
    test of interpolation method d-mean
    """
    # test with regular grid and 1d coords
    grid_lon = np.arange(100)
    grid_lat = np.arange(50)
    data = np.zeros((50, 100))

    # the four nearest values for the first point
    data[20, 10] = 7

    # the four nearest values for the second point
    data[17, 13] = 8

    # the actual test
    res = enstools.interpolation.nearest_neighbour(grid_lon, grid_lat, (10, 13), (20, 17), npoints=2, method="d-mean")(data)
    np.testing.assert_array_almost_equal(res, [5.6, 6.4])
