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
from .os_support import get_first_free_port, which, getenv


@six.add_metaclass(ABCMeta)
class BatchJob():
    """
    abstract base class for a batch job
    """

    # scheduler and work processes started inside of this job.
    def __init__(self):
        """
        read the environment in initialize some variables.
        """
        self.child_processes = []
        self.ip_address = socket.gethostbyname(socket.gethostname())
        self.client = None
        atexit.register(self.cleanup)

    def start_dask_scheduler(self, local_dir=None):
        """
        the scheduler is started as child of this process
        """
        # get a free port for the new scheduler
        self.scheduler_port = get_first_free_port(self.ip_address, 8786)
        # TODO: check if successful!
        args = [sys.executable, which("dask-scheduler"), "--port", "%d" % self.scheduler_port, "--host", self.ip_address]
        if local_dir is not None:
            args.insert(2, "--local-directory")
            args.insert(3, local_dir)
        p = subprocess.Popen(args)
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
        if self.client is None:
            self.client = distributed.Client("tcp://%s:%d" % (self.ip_address, self.scheduler_port))
        return self.client

    def cleanup(self):
        """
        terminate all children
        """
        # shutdown all workers
        if self.client is not None:
            # get a list of all workers and close them
            workers = list(self.client.scheduler_info()['workers'])
            self.client.sync(self.client.scheduler.retire_workers, workers=workers, close_workers=True)

            # wait for the workers to exit
            for one_child in self.child_processes[1:]:
                one_child.wait()

            # close the connection to the scheduler
            self.client.close()
            self.client = None

        # terminate remaining processes (should only be the scheduler)
        for one_child in self.child_processes:
            if one_child.returncode is None:
                one_child.terminate()

    def __del__(self):
        """
        remove children when this object is removed from memory.
        """
        self.cleanup()


class SlurmJob(BatchJob):
    """
    An interface to the SLURM-job inside of which we are running
    """
    def __init__(self):
        """
        read environment
        """
        super(SlurmJob, self).__init__()
        self.job_id = getenv("SLURM_JOBID")
        self.ntasks = int(getenv("SLURM_NTASKS"))
        self.nodelist = getenv("SLURM_JOB_NODELIST")

    def start_dask_worker(self, local_dir=None):
        """
        use srun to start a worker on every allocated cpu.
        """
        args = ["srun", sys.executable, which("dask-worker"), "tcp://%s:%d" % (self.ip_address, self.scheduler_port)]
        if local_dir is not None:
            # TODO: remove temporal folder on all nodes
            args.insert(3, "--local-directory")
            args.insert(4, local_dir)
        p = subprocess.Popen(args)
        self.child_processes.append(p)


def get_batch_job():
    """
    read several environment variables to figure out whether or not we are inside of a Job script or not.

    Returns
    -------
    BatchJob :
            A batch job object for the detected Job type
    """
    # are we inside of SLURM?
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id is not None:
        return SlurmJob()

    # no scheduler found?
    return None


