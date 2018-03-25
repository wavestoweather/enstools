"""
functions used to create dask-clusters automatically based on the environment a script is executed in.
"""


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


# default scheduler for dask
dask.set_options(get=dask.multiprocessing.get)
dask.set_options(pool=multiprocessing.Pool(get_num_available_procs()))


