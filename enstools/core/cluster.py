"""
functions used to create dask-clusters automatically based on the environment a script is executed in.
"""
import os
import sys
import dask
import distributed
import multiprocessing
from .tempdir import TempDir
from .batchjob import get_batch_job, _get_num_available_procs
import atexit
import logging
from time import sleep

# storage the batchjob object
batchjob_object = None

# adapt some settings for dask
from distributed.config import config
config["connect-timeout"] = "30"        # increase the connect-timeout from 3 to 10s


def init_cluster(ntasks=None):
    """
    Create a Dask.distributed cluster and return the client object. The type of the cluster is automatically selected
    based on the environment of the script. Inside of a SLURM job, a distributed Cluster is created. All allocated
    resources are used for this purpose. Without a job scheduler like SLURM, a LocalCluster is created.

    Parameters
    ----------
    ntasks : int
            the number of tasks (threads or processes) to start.

    Returns
    -------
    distributed.Client
            a client object usable to submit computations to the new cluster.
    """
    # create a temporal directory for the work log files
    tmpdir = TempDir(cleanup=False)

    # figure out which type of cluster to create
    global batchjob_object
    batchjob_object = get_batch_job(local_dir=tmpdir.getpath(), ntasks=ntasks)

    # start the distributed cluster prepared by the command above
    batchjob_object.start()

    return batchjob_object.get_client()


def get_num_available_procs():
    """
    Get the number of processes available for parallel execution of tasks. If a distributed cluster was started before,
    then the number of workers within this cluster is returned. Otherwise the number of physical processors on the local
    computer. If OMP_NUM_THREADS is defined, it's value will be used!

    Returns
    -------
    int :
            number of processors.
    """
    if batchjob_object is not None:
        return batchjob_object.ntasks
    else:
        return _get_num_available_procs()
