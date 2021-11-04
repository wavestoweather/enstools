"""
#
# Functions to compress netcdf/grib files from the command line.
#

"""
from typing import Union, List, Tuple
from os.path import isfile, basename
import time


def fix_filename(file_name):
    cases = [".grib2", ".grb"]
    for case in cases:
        file_name = file_name.replace(case, ".nc")
    return file_name


def transfer(file_paths: List[str], output_folder: str, compression: str = "lossless",
             variables_to_keep: List[str] = None):
    """
    This function loops through a list of files creating delayed dask tasks to copy each one of the files while
    optionally using compression.
    If there are dask workers available the tasks will be automatically distributed when using compute.

    Parameters:
    -----------

    file_paths: list of strings
                A list of all the files that will be copied.
    output_folder: string
                Path to the destination folder
    compression: string
                Compression specification or path to json configuration file.
    variables_to_keep: list of strings
                In case of only wanting to keep certain variables, pass the variables to keep as a list of strings.
    """
    from dask import compute
    # Create and fill the list of tasks
    tasks = []
    for file_path in file_paths:
        new_file_path = destination_path(file_path, output_folder)
        # Create task:
        # The transfer file function returns a write task which hasn't been computed.
        # It is not necessary anymore to use the delayed function.
        task = transfer_file(file_path, new_file_path, compression, variables_to_keep)
        # Add task to the list
        tasks.append(task)

    # Compute all the tasks
    compute(tasks)


def transfer_file(origin: str, destination: str, compression: str, variables_to_keep: List[str] = None):
    """
    This function will copy a dataset while optionally applying compression.

    Parameters:
    ----------
    origin: string
            path to original file that will be copied.

    destination: string
            path to the new file that will be created.

    compression: string
            compression specification or path to json configuration file
    """
    from .reader import read
    from .writer import write
    dataset = read(origin, decode_times=False)
    if variables_to_keep is not None:
        # Drop the undesired variables and keep the coordinates
        coordinates = [v for v in dataset.coords]
        variables = [v for v in dataset.variables if v not in coordinates]
        variables_to_drop = [v for v in variables if v not in variables_to_keep]
        dataset = dataset.drop_vars(variables_to_drop)

    return write(dataset, destination, compression=compression, compute=False)


def destination_path(origin_path: str, destination_folder: str):
    """
    Function to obtain the destination file from the source file and the destination folder.
    If the source file has GRIB format (.grb) , it will be changed to netCDF (.nc).
    
    Parameters
    ----------
    origin_path : string
            path to the original file

    destination_folder : string
            path to the destination folder

    Returns the path to the new file that will be placed in the destination folder.
    """
    from os.path import join, basename, splitext
    from enstools.io.file_type import get_file_type

    file_name = basename(origin_path)

    file_format = get_file_type(file_name, only_extension=True)
    if file_format != "NC":
        bname, _ = splitext(file_name)
        file_name = bname + ".nc"
    destination = join(destination_folder, file_name)
    return destination


def compress(output_folder: str, file_paths: List[str], compression: str, nodes: int = 0,
             variables_to_keep: List[str] = None):
    """
    Copies a list of files to a destination folder, optionally applying compression.
    """

    # In case of using automatic compression option, call here get_compression_parameters()
    if compression == "auto":
        from .analyzer import analyze
        compression_parameters_path = "compression_parameters.yaml"
        # By using thresholds = None we will be using the default values.
        analyze(file_paths, thresholds=None, output_file=compression_parameters_path)
        # Now lets continue setting compression = compression_parameters_path
        compression = compression_parameters_path

    # In case of wanting to use additional nodes
    if nodes > 0:
        from enstools.core import init_cluster
        from dask.distributed import performance_report
        with init_cluster(nodes, extend=True) as client:
            client.wait_for_workers(nodes)
            client.get_versions(check=True)
            with performance_report(filename="dask-report.html"):
                # Transfer will copy the files from its origin path to the output folder,
                # using read and write functions from enstools
                init_time = time.time()
                transfer(file_paths, output_folder, compression, variables_to_keep=variables_to_keep)

    else:
        # Transfer will copy the files from its origin path to the output folder,
        # using read and write functions from enstools
        init_time = time.time()
        transfer(file_paths, output_folder, compression, variables_to_keep=variables_to_keep)

    # We could compute compression ratios
    from .metrics import compression_ratio
    from os.path import join, basename
    from pprint import pprint
    compression_ratios = {}
    for file_path in file_paths:
        file_name = basename(file_path)
        file_name = fix_filename(file_name)
        new_file_path = join(output_folder, file_name)
        print(new_file_path, isfile(new_file_path))
        if isfile(new_file_path):
            CR = compression_ratio(file_path, new_file_path)
            compression_ratios[basename(file_path)] = CR
    print("Compression ratios after compression:")
    pprint(compression_ratios)
    end_time = time.time()
    return end_time - init_time
