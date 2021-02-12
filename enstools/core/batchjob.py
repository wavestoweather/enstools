"""
internally used classes und functions that describe a batch job. This is meant for batchjobs inside of which this python
script is executed
"""
import os
import sys
from abc import ABCMeta, abstractmethod
import six
import traceback
import atexit
import socket
import distributed
import logging
from time import sleep
import shutil
import multiprocessing
import threading
import psutil
from .os_support import get_first_free_port, which, getenv, ProcessObserver, get_ip_address


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
        self.lock = threading.Lock()
        self.child_processes = []
        self.client = None
        self.cluster = None
        self.local_dir = local_dir
        self.ip_address = get_ip_address()
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
            logging.debug("waiting for workers... (%d/%d)" % (len(self.client.scheduler_info()["workers"]), self.ntasks))
            sleep(1)

        # 3rd step: make sure that all workers have the same PYTHONPATH
        def set_python_path(first_element):
            sys.path.insert(0, first_element)
        self.client.run(set_python_path, sys.path[0])
        self.client.run_on_scheduler(set_python_path, sys.path[0])

    def start_local(self):
        """
        start a local cluster. This can be used by different implementation as fallback
        """
        # TODO: the start procedure is not 100% reliable, change that!
        #       Possibly related: https://github.com/dask/distributed/issues/1321

        # set memory limit to 90% of the total system memory
        mem_per_worker = int(self.memory_per_node * 0.9)
        logging.debug("memory limit per worker: %db" % mem_per_worker)

        # create the cluster by starting the client
        self.scheduler_port = get_first_free_port(self.ip_address, 8786)
        # When using a single node, the port and ip options won't be specified.
        """
        self.cluster = distributed.LocalCluster(n_workers=self.ntasks,
                                         local_dir=self.local_dir,
                                         scheduler_port=self.scheduler_port,
                                         ip=self.ip_address,
                                         silence_logs=logging.WARN,
                                         threads_per_worker=1,
                                         memory_limit=mem_per_worker)
        """
        self.cluster = distributed.LocalCluster(n_workers=self.ntasks,
                                         local_dir=self.local_dir,
                                         silence_logs=logging.WARN,
                                         threads_per_worker=1,
                                         memory_limit=mem_per_worker)
        self.client = distributed.Client(self.cluster)
        logging.debug("client and cluster started: %s" % str(self.client))

    def start_dask_scheduler(self):
        """
        the scheduler is started as child of this process
        """
        # get a free port for the new scheduler
        self.scheduler_port = get_first_free_port(self.ip_address, 8786)
        # TODO: check if successful!
        args = [which(sys.executable), which("dask-scheduler"), "--port", "%d" % self.scheduler_port, "--host", self.ip_address]
        logging.debug("Scheduler has been started at port %d and host %s" % (self.scheduler_port, self.ip_address))
        #if self.local_dir is not None:
        #    args.insert(2, "--local-directory")
        #    args.insert(3, self.local_dir)
        p = ProcessObserver(args)
        self.child_processes.append(p)

    @abstractmethod
    def start_dask_worker(self):
        """
        Start a worker for every SLURM-Task
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
                    for sec in range(15):
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
            pass
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
                pass
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
    def __init__(self, local_dir=None, ntasks=None):
        """
        read environment
        """
        super(SlurmJob, self).__init__(local_dir)
        self.job_id = getenv("SLURM_JOBID")
        self.step_id = os.getenv("SLURM_STEP_ID")
        self.ntasks = int(getenv("SLURM_NTASKS"))
        if ntasks is not None:
            if ntasks > self.ntasks:
                logging.warning("more worker requested than the number of tasks within this SLURM job.")
                logging.warning("number of worker automatically reduced to %d" % self.ntasks)
            else:
                self.ntasks = ntasks
        # if we are already inside of a slurm step, we can not start more steps. In this case,
        # do not use srun, but start a local cluster
        if self.step_id is not None:
            self.ip_address = "127.0.0.1"
            self.nnodes = 1
            self.nodelist = socket.gethostname()
        else:
            self.nnodes = int(getenv("SLURM_NNODES"))
            self.nodelist = getenv("SLURM_JOB_NODELIST")
        self.memory_per_node = int(getenv("SLURM_MEM_PER_NODE")) * 1048576    # unit: Byte

    def start(self):
        """
        Start a cluster using srun, but only if we are not inside of a slurm step
        """
        if self.step_id is not None:
            self.start_local()
        else:
            super(SlurmJob, self).start()

    def start_dask_worker(self):
        """
        use srun to start a worker on every allocated cpu.
        """
        # calculate the available memory per worker
        mem_per_worker = (self.memory_per_node * self.nnodes) // self.ntasks
        args = [which("srun"),
                "--ntasks=%d" % self.ntasks,
                sys.executable, which("dask-worker"),
                "--nthreads", "1",
                "--memory-limit", "%d" % mem_per_worker,
                "tcp://%s:%d" % (self.ip_address, self.scheduler_port)
                ]
        logging.debug(" ".join(args))
        #if self.local_dir is not None:
        #    args.insert(4, "--local-directory")
        #    args.insert(5, self.local_dir)
        p = ProcessObserver(args)  # subprocess.Popen(args)
        self.child_processes.append(p)

        # specify a cleanup command to execute after the workers finished
        if self.local_dir is not None:
            p.run_on_exit([which("srun"), "--ntasks=%d" % self.nnodes, "--ntasks-per-node=1", which("rm"), "-rf", self.local_dir])


class LocalJob(BatchJob):
    """
    start a LocalCluster instead of a batch job cluster
    """
    def __init__(self, local_dir=None, ntasks=None):
        super(LocalJob, self).__init__(local_dir)
        self.ntasks = _get_num_available_procs()
        if ntasks is not None:
            if ntasks > self.ntasks:
                logging.warning("More worker requested then CPUs available.")
                logging.warning("Since we are running on a local computer without job scheduler, they are started anyway.")
            self.ntasks = ntasks
        self.ip_address = "127.0.0.1"
        self.nnodes = 1
        self.nodelist = socket.gethostname()
        self.memory_per_node = psutil.virtual_memory().total

    def start_dask_worker(self):
        pass

    def start(self):
        """
        Start a local cluster
        """
        self.start_local()

    def cleanup(self):
        self.lock.acquire()
        try:
            # close the cluster and the connection to the cluster
            if self.client is not None:
                # if there is still something running on the cluster, cancel it
                data_refs = self.client.has_what()
                for one_worker, keys in six.iteritems(data_refs):
                    for key in keys:
                        f = distributed.Future(key, client=self.client)
                        self.client.cancel(f)

                # close the client and the cluster itself
                self.client.close()
                self.client = None
                workers = list(self.cluster.workers)
                for one_worker in workers:
                    self.cluster.stop_worker(one_worker)
                self.cluster.close()

            # remove the temporal folder on the local host
            if self.local_dir is not None and os.path.exists(self.local_dir):
                shutil.rmtree(self.local_dir)
                self.local_dir = None
        except:
            raise
        finally:
            self.lock.release()


def get_batch_job(local_dir=None, **kwargs):
    """
    read several environment variables to figure out whether or not we are inside of a Job script or not.

    Parameters
    ----------
    local_dir : str
            absolute path of a folder used as temporal directory

    **kwargs:
            all arguments are passed on the the init function of the different job implementations

    Returns
    -------
    BatchJob :
            A batch job object for the detected Job type
    """
    # are we inside of SLURM?
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id is not None:
        # Check if we are in more than one node:
        node_list = os.getenv("SLURM_NODELIST")
        logging.debug("Node list: %s" % node_list)
        number_of_nodes = int(os.getenv("SLURM_NNODES"))
        logging.debug("Number of nodes in the current slurm job: %d" % number_of_nodes)
        if number_of_nodes > 1:
            
            return SlurmJob(local_dir, **kwargs)

    # no job environment detected? create a local cluster!
    return LocalJob(local_dir, **kwargs)

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
