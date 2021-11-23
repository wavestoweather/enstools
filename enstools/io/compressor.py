"""
#
# Functions to compress netcdf/grib files from the command line.
#

"""
from typing import Union, List, Tuple
from os.path import isfile, isdir, basename
from os import rename
import time


def fix_filename(file_name):
    cases = [".grib2", ".grb"]
    for case in cases:
        file_name = file_name.replace(case, ".nc")
    return file_name


def transfer(file_paths: Union[List[str], str], output: str, compression: str = "lossless",
             variables_to_keep: List[str] = None):
    """
    This function loops through a list of files creating delayed dask tasks to copy each one of the files while
    optionally using compression.
    If there are dask workers available the tasks will be automatically distributed when using compute.

    Parameters:
    -----------

    file_paths: string or list of strings
                File-path or list of file-paths of all the files that will be copied.
    output_folder: string
                Path to the destination folder
    compression: string
                Compression specification or path to json configuration file.
    variables_to_keep: list of strings
                In case of only wanting to keep certain variables, pass the variables to keep as a list of strings.
    """
    # If its a single file, just create a list with it.
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # If we have a single file, we might accept a output filename instead of an output folder.
    # Some assertions first to prevent wrong usage.
    if len(file_paths) == 0:
        raise AssertionError("file_paths can't be an empty list")
    elif len(file_paths) == 1:
        file_path = file_paths[0]
        new_file_path = destination_path(file_path, output) if isdir(output) else output
        transfer_file(file_path, new_file_path, compression, variables_to_keep)
    elif len(file_paths) > 1:
        # In case of having more than one file, check that output corresponds to a directory
        assert isdir(output), "For multiple files, the output parameter should be a directory"
        transfer_multiple_files(
                                file_paths=file_paths,
                                output=output,
                                compression=compression,
                                variables_to_keep=variables_to_keep,
                                )


def transfer_multiple_files(file_paths: Union[List[str],str], output: str, compression: str = "lossless",
                            variables_to_keep: List[str] = None):
    from dask import compute
    # Create and fill the list of tasks
    tasks = []

    # In order to give the files a temporary filename we will use a dictionary to keep the old and new names
    temporary_names_dictionary = {}

    for file_path in file_paths:
        new_file_path = destination_path(file_path, output)

        # Create temporary file name and store it in the dictionary
        temporary_file_path = f"{new_file_path}.tmp"
        temporary_names_dictionary[temporary_file_path] = new_file_path

        # Create task:
        # The transfer file function returns a write task which hasn't been computed.
        # It is not necessary anymore to use the delayed function.
        task = transfer_file(file_path, temporary_file_path, compression, variables_to_keep, compute=False)
        # Add task to the list
        tasks.append(task)

    # Compute all the tasks
    compute(tasks)

    # Rename files
    for temporary_name, final_name in temporary_names_dictionary.items():
        rename(temporary_name, final_name)


def transfer_file(origin: str, destination: str, compression: str, variables_to_keep: List[str] = None,
                  compute: bool = True):
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

    return write(dataset, destination, file_format="NC", compression=compression, compute=compute)


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


def compress(file_paths: Union[List[str], str], output: str, compression: str, nodes: int = 0,
             variables_to_keep: List[str] = None):
    """
    Copies a list of files to a destination folder, optionally applying compression.
    """

    # In case of using automatic compression option, call here get_compression_parameters()
    if compression == "auto":
        from .analyzer import analyze
        import os
        compression_parameters_path = os.path.join(output, "compression_parameters.yaml")
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
                transfer(file_paths, output, compression, variables_to_keep=variables_to_keep)

    else:
        # Transfer will copy the files from its origin path to the output folder,
        # using read and write functions from enstools
        init_time = time.time()
        transfer(file_paths, output, compression, variables_to_keep=variables_to_keep)

    end_time = time.time()

    check_compression_ratios(file_paths, output)

    return end_time - init_time


def check_compression_ratios(file_paths: Union[List[str], str], output: str):
    # Compute compression ratios
    from .metrics import compression_ratio
    from os.path import join, basename
    from pprint import pprint
    compression_ratios = {}

    # Single file case
    if isinstance(file_paths, str):
        file_path = file_paths
        if isdir(output):
            file_name = basename(file_path)
            new_file_path = join(output, file_name)
        elif isfile(output):
            new_file_path = output
        CR = compression_ratio(file_path, new_file_path)
        print(f"Compression ratios after compression:\nCR: {CR:.1f}")
        return

    # Multiple files
    for file_path in file_paths:
        file_name = basename(file_path)
        file_name = fix_filename(file_name)
        new_file_path = join(output, file_name)
        print(new_file_path, isfile(new_file_path))
        if isfile(new_file_path):
            CR = compression_ratio(file_path, new_file_path)
            compression_ratios[basename(file_path)] = CR
    print("Compression ratios after compression:")
    pprint(compression_ratios)

