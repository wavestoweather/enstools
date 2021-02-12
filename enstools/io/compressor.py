"""
#
# Functions to compress netcdf/grib files from the command line.
#

"""

def init_cluster(nodes=1):
    """
    # Submiting DASK workers to a Slurm cluster. Need to merge it with the init_cluster in enstools.core

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
    import glob
    import argparse
    from os.path import isdir, realpath
    from os import access, W_OK

    def expand_paths(string):
        """
        Small function to expand the file paths
        """
        files = glob.glob(string)
        return [realpath(f) for f in files]
    
    help_text = """
Few different examples

-Single file:
%(prog)s -o /path/to/destination/folder /path/to/file/1 

-Multiple files:
%(prog)s -o /path/to/destination/folder /path/to/file/1 /path/to/file/2

-Path pattern:
%(prog)s -o /path/to/destination/folder /path/to/multiple/files/* 

Launch a SLURM job for workers:
%(prog)s -o /path/to/destination/folder /path/to/multiple/files/* --nodes 4

To use custom compression parameters:
%(prog)s -o /path/to/destination/folder /path/to/multiple/files/* --compression compression_specification

Where compression_specification can contain the string that defines the compression options that will be used in the whole dataset,
        or a filepath to a configuration file in which we might have per variable specification.
        For lossless compression, we can choose the backend and the compression leven as follows
            "lossless:backend:compression_level(from 1 to 9)"
        The backend must be one of the following options:
                'blosclz'
                'lz4'
                'lz4hc'
                'snappy'
                'zlib'
                'zstd'
        For lossy compression, we can choose the compressor (wight now only zfp is implemented),
        the method and the method parameter (the accuracy, the precision or the rate).
        Some examples:
            "lossless"
            "lossy"
            "lossless:zlib:5"
            "lossy:zfp:accuracy:0.00001"
            "lossy:zfp:precision:12"
            
        If using a configuration file, the file should follow a json format and can contain per-variable values.
        It is also possible to define a default option. For example:
        { "default": "lossless",
          "temp": "lossy:zfp:accuracy:0.1",
          "qv": "lossy:zfp:accuracy:0.00001"        
        }


So, few examples with custom compression would be:

%(prog)s -o /path/to/destination/folder /path/to/multiple/files/* --compression lossless

%(prog)s -o /path/to/destination/folder /path/to/multiple/files/* --compression lossless:blosclz:9

%(prog)s -o /path/to/destination/folder /path/to/multiple/files/* --compression lossy

%(prog)s -o /path/to/destination/folder /path/to/multiple/files/* --compression lossy:zfp:rate:4

%(prog)s -o /path/to/destination/folder /path/to/multiple/files/* --compression compression_paramters.json


Last but not least, now it is possible to automatically find which are the compression parametrs that must be applied to each variable in order to mantain a 0.99999 Pearson corre

%(prog)s -o /path/to/destination/folder /path/to/multiple/files/* --compression auto


"""

    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawDescriptionHelpFormatter)

    #parser = OptionParser(usage=help_text)
    parser.add_argument("files", type=expand_paths, nargs='*', help="Path to file/files that will be compressed")
    parser.add_argument("-o", '--output-folder', type=str, dest="output_folder", default=None, required=True)
    parser.add_argument('--compression', type=str, dest="compression", default="lossless",
                      help="""
        Specifications about the compression options that will be used in the whole dataset,
        or a filepath to a configuration file in which we might have per variable specification.
        Some examples:   
            "lossless"
            "lossy" 
            "lossless:zlib:5"
            "lossy:zfp:accuracy:0.00001"
            "lossy:zfp:precision:12"
            "configuration_file.json"
        If using a configuration file, the file should follow a json format and can contain per-variable values.
        It is also possible to define a default option. For example:
        { "default": "lossless",
          "temp": "lossy:zfp:accuracy:0.1",
          "qv": "lossy:zfp:accuracy:0.00001"        
        }""")
    parser.add_argument("--nodes", "-N", dest="nodes", default=0, type=int,
                     help="This parameter can be used to allocate additional nodes in the cluster to speed-up the computation.")


    args = parser.parse_args()
    
    # Read the output folder from the command line and assert that it exists and has write permissions.
    output_folder = realpath(args.output_folder)
    assert isdir(output_folder), "The provided folder does not exist"
    assert access(output_folder,W_OK ), "The output folder provided does not have write permissions"

    file_paths = args.files
    file_paths = sum(file_paths, [])

    # Compression options
    compression = args.compression
    # Read the number of nodes
    nodes = args.nodes
        
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





def launch_compress_from_command_line():
    """
    Read command line arguments and launch the compress function
    """
    # Parse command line arguments
    output_folder, file_paths, compression, nodes = parse_command_line_arguments()
    compress(output_folder, file_paths, compression, nodes)

def compress(output_folder, file_paths, compression, nodes):
    """
    Copies a list of files to a destination folder, optionally applying compression.
    """
    
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
    launch_compress_from_command_line()
