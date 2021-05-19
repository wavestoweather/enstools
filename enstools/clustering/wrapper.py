"""
Wrapper around sklearn cluster methods for more convenience.
"""
import sklearn.cluster
from sklearn.metrics import silhouette_score
import numpy as np
import xarray
import dask.delayed
import dask.array

methods = {"kmeans": sklearn.cluster.KMeans,
           "aprop": sklearn.cluster.AffinityPropagation,
           "mshift": sklearn.cluster.MeanShift,
           "spectral": sklearn.cluster.SpectralClustering,
           "agglo": sklearn.cluster.AgglomerativeClustering,
           "dbscan": sklearn.cluster.DBSCAN,
           "birch": sklearn.cluster.Birch}

methods_with_ncluster = {
           "kmeans": True,
           "aprop": False,
           "mshift": False,
           "spectral": True,
           "agglo": True,
           "dbscan": False,
           "birch": True}


def cluster(algorithm, data, n_clusters=None, n_clusters_max=None, sort=True, **kwargs):
    """

    Parameters
    ----------
    algorithm : str
            clustering method to use. all sklearn methods are supported. $methods

    data : xarray.DataArray or np.ndarray
            the input data for the clustering. This is the output of the :func:`prepare` function.

    n_clusters : int or None
            number of clusters to create. An Integer will create the specified number of
            clusters. None will try to estimate the number of clusters using the silhouette score if the algorithm
            supports the prescription of the number of clusters.

    n_clusters_max : int
            maximal number of clusters to create. The default is the number of ensemble members divided by four.

    sort : bool
            If True, the clusters are sorted in order to create better reproducible results.

    **kwargs
            all

    Returns
    -------
    np.ndarray
            1d array with cluster labels.
    """

    # is a valid algorithm selected?
    if algorithm not in methods:
        raise ValueError("unsupported algorithm selected: %s supported are only: %s" % (algorithm, ", ".join(methods.keys())))

    # is it possible to prescribe the number of clusters?
    has_n_clusters = methods_with_ncluster[algorithm]

    # estimate the number of clusters?
    scores = None
    if has_n_clusters and n_clusters is None:
        # calculate clustering up the 1/3 times the ensemble members
        n_clusters_min = 2
        if n_clusters_max is None:
            n_clusters_max = data.shape[0] // 4

        labels = []
        scores = []
        for nc in range(n_clusters_min, n_clusters_max + 1):
            labels.append(dask.delayed(cluster)(algorithm, data, n_clusters=nc, sort=False, **kwargs))
        labels = dask.compute(*labels)
        for nc in range(n_clusters_min, n_clusters_max + 1):
            scores.append(dask.delayed(silhouette_score)(data, np.asarray(labels[nc - n_clusters_min])))

        # the the number of clusters with the largest score
        scores = np.array(dask.compute(*scores))
        n_clusters = scores.argmax() + n_clusters_min
        result = labels[scores.argmax()]

    # number of clusters is given as argument
    elif has_n_clusters:
        n_clusters_min = n_clusters
        n_clusters_max = n_clusters
        model = methods[algorithm](n_clusters=n_clusters, **kwargs).fit(data)
        result = model.labels_

    # sort the result?
    if sort:
        # calculate the variance for each cluster
        cluster_in_res = np.unique(result)
        cluster_var = list(map(lambda x: data[np.asarray(result) == x, ...].var(axis=0).mean(), cluster_in_res))

        # sort the clusters by their variance between the members
        sorted_clusters = sorted(zip(cluster_in_res, cluster_var), key=lambda x: x[1])

        # relabel the result
        for i_cluster, one_cluster in enumerate(sorted_clusters):
            result = np.where(result == one_cluster[0], i_cluster + len(cluster_in_res), result)
        result -= len(cluster_in_res)

    # convert to xarray and add some meta data
    result = xarray.DataArray(result, dims=("ens",))
    result.attrs["n_clusters"] = int(result.max() + 1)
    if has_n_clusters:
        result.attrs["n_clusters_min"] = n_clusters_min
        result.attrs["n_clusters_max"] = n_clusters_max
    result.attrs["algorithm"] = algorithm
    if scores is not None:
        result.attrs["scores"] = scores

    return result


# add a list of methods to the doc-string
__methods_str = "\n\n"
for __method_name in sorted(methods.keys()):
    __methods_str += "            *%s*: :class:`sklearn.cluster.%s`\n\n" % (__method_name, methods[__method_name].__name__)
cluster.__doc__ = cluster.__doc__.replace("$methods", __methods_str)
