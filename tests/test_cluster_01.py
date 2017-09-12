import nose
import numpy
from enstools.cluster import prepare
from sklearn.cluster import KMeans

variables = []
ens_members = 10
n_variables = 2


def setup():
    """
    create two variables for clustering
    """

    for ivar in range(n_variables):
        var = numpy.random.randn(ens_members, 10, 10)
        for iens in range(ens_members):
            if iens % 2 == 0:
                var[iens, 0:5, :] += 1
            else:
                var[iens, 5:10, :] += 1
        variables.append(var)


def test_prepare():
    """
    test of the variable preparation enstools.cluster.prepare
    """
    # input with different shapes
    with numpy.testing.assert_raises(ValueError):
        x = prepare(numpy.zeros((ens_members, 10, 11)), *variables)

    # test with valid input
    x = prepare(*variables)
    numpy.testing.assert_array_equal(x.shape, (ens_members, 200))


def test_prepare_kmeans():
    """
    use the prepare function to perform a kmeans clustering
    """
    # prepare the data
    x = prepare(*variables)

    # perform the clustering
    labels = KMeans(n_clusters=2).fit_predict(x)
    # all even elements should be in one cluster, all odd in another
    even = labels[0:ens_members:2]
    odd = labels[1:ens_members:2]
    numpy.testing.assert_array_equal(even, numpy.repeat(even[0], ens_members / 2))
    numpy.testing.assert_array_equal(odd, numpy.repeat(odd[0], ens_members / 2))
    numpy.testing.assert_equal(even[0] != odd[0], True)
