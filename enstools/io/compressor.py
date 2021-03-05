"""
#
# Functions to compress netcdf/grib files from the command line.
#

"""


def transfer(file_paths, output_folder, compression="lossless", variables_to_keep=None):
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
    """
    from dask import delayed, compute
    # Create the delayed version of the transfer_file function
    delayed_transfer_file = delayed(transfer_file)
    # Create and fill the list of delayed tasls
    tasks = []
    for file_path in file_paths:
        new_file_path = destination_path(file_path, output_folder)
        # Create task
        task = delayed_transfer_file(file_path, new_file_path, compression, variables_to_keep)
        # Add task to the list
        tasks.append(task)

    # Compute all the tasks
    compute(tasks)

    
def transfer_file(origin, destination, compression, variables_to_keep=None):
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
    import hdf5plugin
    dataset = read(origin, decode_times=False)
    if variables_to_keep is not None:
        # Drop the undesired variables and keep the coordinates
        coordinates = [v for v in dataset.coords]
        variables = [v for v in dataset.variables if v not in coordinates]
        variables_to_drop = [v for v in variables if v not in variables_to_keep]
        dataset = dataset.drop_vars(variables_to_drop)

    write(dataset, destination, compression=compression)


def destination_path(origin_path, destination_folder):
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
    from os.path import join, basename
    destination = join(destination_folder, basename(origin_path))
    if destination.count(".grb"):
        destination = destination.replace(".grb", ".nc")
    return destination


def compress(output_folder, file_paths, compression, nodes, variables_to_keep=None):
    """
    Copies a list of files to a destination folder, optionally applying compression.
    """
    
    # In case of using automatic compression option, call here get_compression_parameters()
    if compression == "auto":
        from .analyzer import analyze
        compression_parameters_path = "compression_parameters.json"
        analyze(file_paths, correlation_threshold=0.99999, output_file=compression_parameters_path)
        # Now lets continue setting compression = compression_parameters_path
        compression = compression_parameters_path
    
    # In case of wanting to use additional nodes
    if nodes > 0:
        from enstools.core import init_cluster
        with init_cluster(nodes, extend=True) as client:
            client.wait_for_workers(nodes)
            # Transfer will copy the files from its origin path to the output folder,
            # using read and write functions from enstools
            transfer(file_paths, output_folder, compression)
    else:
        # Transfer will copy the files from its origin path to the output folder,
        # using read and write functions from enstools
        transfer(file_paths, output_folder, compression, variables_to_keep=variables_to_keep)
