"""
functions used to create dask-clusters automatically based on the environment a script is executed in.
"""
import os
import sys
import dask
import distributed
import multiprocessing
from .tempdir import TempDir
from .batchjob import get_batch_job
import atexit
import logging
from time import sleep

# storage for client and batchjob objects
client_object = None
batchjob_object = None


def get_num_available_procs():
    """
    Find the number of processors available for computations. The function will look for environment variables from
    SLURM and openMP (in that order). If no environment variables are defined, the number of cpus will be returned.

    Returns
    -------
    int
            Number of processes available for parallel computations.
    """
    # running inside a slurm job?
    SLURM_NTASKS = os.getenv("SLURM_NTASKS")
    if SLURM_NTASKS is not None:
        return int(SLURM_NTASKS)

    # openmp number of threads?
    OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS")
    if OMP_NUM_THREADS is not None:
        return int(OMP_NUM_THREADS)

    # no hints from environment variables, the number of hardware cpus is returned
    return multiprocessing.cpu_count()


def init_cluster():
    """
    Create a Dask.distributed cluster and return the client object. The type of the cluster is automatically selected
    based on the environment of the script. Inside of a SLURM job, a distributed Cluster is created. All allocated
    resources are used for this purpose. Without a job scheduler like SLURM, a LocalCluster is created.

    Returns
    -------
    distributed.Client
            a client object usable to submit computations to the new cluster.
    """
    # create a temporal directory for the work log files
    tmpdir = TempDir(cleanup=False)

    # figure out which type of cluster to create
    global batchjob_object
    batchjob_object = get_batch_job()
    if batchjob_object is None:
        # no job found. start a LocalCluster
        logging.error("LocalCluster: not yet implemented!")
        # create the init_cluster, which will create the cluster as well
        client = distributed.Client(local_dir=tmpdir.getpath())
    else:
        # use the scheduler to start a distributed cluster
        # 1st step: start a local scheduler and remote workers
        batchjob_object.start_dask_scheduler(tmpdir.getpath())
        batchjob_object.start_dask_worker(tmpdir.getpath())

        # 2nd step: connect to the new dask cluster
        client = batchjob_object.get_client()
        # wait until all workers are started and connected to the scheduler
        while len(client.scheduler_info()["workers"]) < batchjob_object.ntasks:
            sleep(1)

        # 3rd step: make sure that all workers have the same PYTHONPATH
        def set_python_path(first_element):
            sys.path.insert(0, first_element)
        client.run(set_python_path, sys.path[0])

    # init_cluster up temporal files at exit
    def clientup_temporal_data():
        # remove temporal files
        tmpdir.cleanup()
    atexit.register(clientup_temporal_data)
    return client

# default scheduler for dask
#dask.set_options(get=dask.multiprocessing.get)
#dask.set_options(pool=multiprocessing.Pool(get_num_available_procs()))


