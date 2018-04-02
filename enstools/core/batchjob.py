"""
internally used classes und functions that describe a batch job. This is meant for batchjobs inside of which this python
script is executed
"""
import os
import sys
from abc import ABCMeta, abstractmethod
import six
import subprocess
import atexit
import socket
import distributed
import logging
from time import sleep
import shutil
import multiprocessing
import threading
from .os_support import get_first_free_port, which, getenv, ProcessObserver


@six.add_metaclass(ABCMeta)
class BatchJob():
    """
    abstract base class for a batch job
    """

    # scheduler and work processes started inside of this job.
    def __init__(self, local_dir=None):
        """
        read the environment in initialize some variables.
        """
        self.child_processes = []
        self.ip_address = socket.gethostbyname(socket.gethostname())
        self.client = None
        self.local_dir = local_dir
        self.lock = threading.Lock()
        atexit.register(self.cleanup)

    def start(self):
        """
        start the scheduler and the worker processes
        """
        # 1st step: start processes
        self.start_dask_scheduler()
        self.start_dask_worker()
        self.client = distributed.Client("tcp://%s:%d" % (self.ip_address, self.scheduler_port))

        # 2nd step: wait until all workers are started and connected to the scheduler
        while len(self.client.scheduler_info()["workers"]) < self.ntasks:
            sleep(1)

        # 3rd step: make sure that all workers have the same PYTHONPATH
        def set_python_path(first_element):
            sys.path.insert(0, first_element)
        self.client.run(set_python_path, sys.path[0])

    def start_dask_scheduler(self):
        """
        the scheduler is started as child of this process
        """
        # get a free port for the new scheduler
        self.scheduler_port = get_first_free_port(self.ip_address, 8786)
        # TODO: check if successful!
        args = [sys.executable, which("dask-scheduler"), "--port", "%d" % self.scheduler_port, "--host", self.ip_address]
        if self.local_dir is not None:
            args.insert(2, "--local-directory")
            args.insert(3, self.local_dir)
        p = ProcessObserver(args)
        self.child_processes.append(p)

    @abstractmethod
    def start_dask_worker(self):
        """
        Start a worker for ever SLURM-Task
        """
        pass

    def get_client(self):
        """
        if not already running, a new client is started and returned
        """
        return self.client

    def cleanup(self):
        """
        terminate all children
        """
        # cleanup is called more then once. The reason is, that it is called by __del__ in case of a normal exit and
        # also by atexit to ensure cleanup in case of a exit by exception.
        self.lock.acquire()
        try:
            # shutdown all workers
            if self.client is not None:
                # get a list of all workers
                workers = list(self.client.scheduler_info()['workers'])

                # if there is still something running on the cluster, cancel it
                data_refs = self.client.has_what()
                for one_worker, keys in six.iteritems(data_refs):
                    for key in keys:
                        f = distributed.Future(key, client=self.client)
                        self.client.cancel(f)

                # stop the workers
                self.client.sync(self.client.scheduler.retire_workers, workers=workers, close_workers=True)

                # wait for the workers to exit
                for one_child in self.child_processes[1:]:
                    for sec in range(5):
                        one_child.poll()
                        if one_child.returncode is not None:
                            break
                        else:
                            sleep(1)
                    if one_child.returncode is None:
                        logging.warning("At least one dask worker didn't quit voluntarily!")

                # close the connection to the scheduler
                self.client.close()
                self.client = None
        except:
            raise
        finally:
            try:
                # terminate remaining processes (should only be the scheduler)
                for one_child in self.child_processes:
                    if one_child.returncode is None:
                        one_child.terminate()

                # remove the temporal folder on the local host
                if self.local_dir is not None and os.path.exists(self.local_dir):
                    shutil.rmtree(self.local_dir)
                    self.local_dir = None
            except:
                raise
            finally:
                self.lock.release()

    def __del__(self):
        """
        remove children when this object is removed from memory.
        """
        self.cleanup()


class SlurmJob(BatchJob):
    """
    An interface to the SLURM-job inside of which we are running
    """
    def __init__(self, local_dir=None):
        """
        read environment
        """
        super(SlurmJob, self).__init__(local_dir)
        self.job_id = getenv("SLURM_JOBID")
        self.ntasks = int(getenv("SLURM_NTASKS"))
        self.nnodes = int(getenv("SLURM_NNODES"))
        self.nodelist = getenv("SLURM_JOB_NODELIST")

    def start_dask_worker(self):
        """
        use srun to start a worker on every allocated cpu.
        """
        args = ["srun", sys.executable, which("dask-worker"), "--nthreads", "1", "tcp://%s:%d" % (self.ip_address, self.scheduler_port)]
        if self.local_dir is not None:
            args.insert(3, "--local-directory")
            args.insert(4, self.local_dir)
        p = ProcessObserver(args)  # subprocess.Popen(args)
        self.child_processes.append(p)

        # specify a cleanup command to execute after the workers finished
        if self.local_dir is not None:
            p.run_on_exit(["srun", "--ntasks=%d" % self.nnodes, "--ntasks-per-node=1", "rm", "-rf", self.local_dir])


class LocalJob(BatchJob):
    """
    start a LocalCluster instead of a batch job cluster
    """
    def __init__(self, local_dir=None):
        super(LocalJob, self).__init__(local_dir)
        self.ntasks = _get_num_available_procs()
        self.ip_address = "127.0.0.1"
        self.nnodes = 1
        self.nodelist = socket.gethostname()

    def start_dask_worker(self):
        pass

    def start(self):
        """
        start a local cluster

        Parameters
        ----------
        local_dir: str
                absolute path to temporal folder
        """
        # TODO: the start procedure is not 100% reliable, change that!
        self.scheduler_port = get_first_free_port(self.ip_address, 8786)
        self.client = distributed.Client(n_workers=self.ntasks,
                                         local_dir=self.local_dir,
                                         scheduler_port=self.scheduler_port,
                                         ip=self.ip_address,
                                         silence_logs=logging.WARN,
                                         threads_per_worker=1)
        logging.debug("client and cluster started: %s" % str(self.client))

    def cleanup(self):
        self.lock.acquire()
        try:
            # close the cluster and the connection to the cluster
            if self.client is not None:
                # close the client and the cluster itself
                self.client.close()
                self.client = None

            # remove the temporal folder on the local host
            if self.local_dir is not None and os.path.exists(self.local_dir):
                shutil.rmtree(self.local_dir)
                self.local_dir = None
        except:
            raise
        finally:
            self.lock.release()


def get_batch_job(local_dir=None):
    """
    read several environment variables to figure out whether or not we are inside of a Job script or not.

    Parameters
    ----------
    local_dir : str
            absolute path of a folder used as temporal directory

    Returns
    -------
    BatchJob :
            A batch job object for the detected Job type
    """
    # are we inside of SLURM?
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id is not None:
        return SlurmJob(local_dir)

    # no job environment detected? create a local cluster!
    return LocalJob(local_dir)

    # no scheduler found?
    raise ValueError("unable to create a new dask cluster!")


def _get_num_available_procs():
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
