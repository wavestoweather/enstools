#!/usr/bin/env python
"""
    #
    # Prototype of the command line tool to compress datasets.
    #

"""

def init_cluster(nodes=1):
    """
    # Submiting DASK workers to a Slurm cluster.

           Parameters
    ----------
    nodes : int
            number of nodes
    """
    from dask_jobqueue import SLURMCluster

    # Define the kind of jobs that will be launched to the cluster
    # This will apply for each one of the different jobs sent
    cluster = SLURMCluster(
        cores=12,
        memory="24 GB",
        queue="cluster",
        local_directory="/dev/shm",
        silence_logs="debug",
    )
    # Start workers
    cluster.scale(jobs=nodes)
    return cluster


def init_client(cluster):
    """
    Init Dask client using a cluster
    """
    from dask.distributed import Client
    # Start client and print link to dashboard
    client = Client(cluster)
    print("You can follow the dashboard in the following link:\n%s" % client.dashboard_link)
    return client


def transfer(file_paths, output_folder, compression="lossless"):
    """
    This function loops through a list of files creating delayed dask tasks to copy each one of the files while optionally using compression. If there are dask workers available the tasks will be automatically distributed when using compute.

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
        task = delayed_transfer_file(file_path, new_file_path, compression)
        # Add task to the list
        tasks.append(task)

    # Compute all the tasks
    compute(tasks)

    
def transfer_file(origin, destination, compression):    
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
    from .encoding import set_encoding
    import hdf5plugin
    dataset = read(origin)
    encoding = set_encoding(dataset, compression)
    write(dataset, destination, compression=compression)

    
def parse_command_line_arguments():
    """
    Parse the command line arguments and return the list of files, the destination folder and the number of nodes that will be used.
    """
    from os import access, W_OK
    from os.path import isdir
    import glob
    from optparse import OptionParser
    
    help_text = """
Few different examples

-Single file:
%prog -o /path/to/destination/folder /path/to/file/1 

-Multiple files:
%prog -o /path/to/destination/folder /path/to/file/1 /path/to/file/2

-Path pattern:
%prog -o /path/to/destination/folder /path/to/multiple/files/* 

Launch a SLURM job for workers:
%prog -o /path/to/destination/folder /path/to/multiple/files/* --nodes 4

Use lossy compression:
%prog -o /path/to/destination/folder /path/to/multiple/files/* --lossy

"""
    
    parser = OptionParser(usage=help_text)
    
    parser.add_option("-o", "--output-folder", dest="output_folder",
                      default=None, help="Path to the output folder")
    parser.add_option("--compression", dest="compression", default="lossless",
                     help="Compression options. Using lossless compression by default. Full description of available options upcoming.")
    parser.add_option("--nodes", "-N", dest="nodes", default=0,
                     help="This parameter can be used to allocate additional nodes in the cluster to speed-up the computation.")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        print("No argument given!")
        parser.print_help()
        exit(1)
    
    # Read the output folder from the command line and assert that it exists and has write permissions.
    output_folder = options.output_folder
    if output_folder is None:
        parser.print_help()
        print("\nThe parameter output folder is mandatory.")
        exit(1)
    assert isdir(output_folder), "The provided folder does not exist"
    assert access(output_folder,W_OK ), "The output folder provided does not have write permissions"

    # Expand the filenames in case we ued a regex expression
    assert len(args) > 0, "Need to provide the list of files as arguments. Regex patterns are allowed."
    file_paths = sum([glob.glob(arg) for arg in args], [])

    # Compression options
    compression = options.compression
    # Read the number of nodes
    nodes = int(options.nodes)
        
    # If we are not using MPI, just return the output folder and the list of files
    return output_folder, file_paths, compression, nodes


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


def compress():
    """
    Copies a list of files provided as command line arguments to a destination folder, optionally applying compression.
    The list of files, the destination folder and the compression options have to be provided as command line options.
    """
    # Parse command line arguments
    output_folder, file_paths, compression, nodes = parse_command_line_arguments()
    
    # In case of using automatic compression option, call here get_compression_parameters()
    if compression == "auto":
        from .analyzer import get_compression_parameters
        compression_parameters_path = "compression_parameters.json"
        get_compression_parameters(file_paths, correlation_threshold=0.99999, output_file=compression_parameters_path)
        # Now lets continue setting compression = compression_parameters_path
        compression = compression_parameters_path
    
    # In case of wanting to use additional nodes
    if nodes > 0:
        with init_cluster(nodes) as cluster, init_client(cluster) as client:
            client.wait_for_workers(nodes)
            # Transfer will copy the files from its origin path to the output folder, using read and write functions from enstools 
            transfer(file_paths, output_folder, compression)
    else:
        # Transfer will copy the files from its origin path to the output folder, using read and write functions from enstools 
        transfer(file_paths, output_folder,compression)

        
if __name__ == "__main__":
    compress()
