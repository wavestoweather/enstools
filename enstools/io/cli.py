"""
Single access for different enstools-compressor utilities
"""

# Few help messages
compressor_help_text = """
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


def expand_paths(string):
    import glob
    from os.path import realpath
    """
    Small function to expand the file paths
    """
    files = glob.glob(string)
    return [realpath(f) for f in files]


def main():
    # Create parser
    import argparse

    # Create the top-level parser
    parser = argparse.ArgumentParser()
    parser.set_defaults(which=None)
    subparsers = parser.add_subparsers(help='Select between the different enstools utilities')

    # Create the parser for the "compressor" command
    parser_compressor = subparsers.add_parser('compress', help='Compress help',
                                              formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description=compressor_help_text)
    parser_compressor.add_argument("files", type=expand_paths, nargs='*',
                                   help="Path to file/files that will be compressed."
                                        "Multiple files and regex patterns are allowed.")
    parser_compressor.add_argument("-o", '--output-folder', type=str, dest="output_folder", default=None, required=True)
    parser_compressor.add_argument('--compression', type=str, dest="compression", default="lossless",
                                   help="""
        Specifications about the compression options. Default is: %(default)s""")
    parser_compressor.add_argument("--nodes", "-N", dest="nodes", default=0, type=int,
                                   help="This parameter can be used to allocate additional nodes in the cluster to speed-up the computation.")

    parser_compressor.set_defaults(which='compressor')

    # Create the parser for the "analyzer" command
    parser_analyzer = subparsers.add_parser('analyze', help='Analyze help',
                                            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser_analyzer.add_argument("--correlation", dest="correlation", default=0.99999, type=float,
                                 help="Correlation threshold. Default=%(default)s")
    parser_analyzer.add_argument("--output", "-o", dest="output", default=None, type=str,
                                 help="Path to the file where the configuration will be saved."
                                      "If not provided will be print in the stdout.")
    parser_analyzer.add_argument("files", type=str, nargs="+",
                                 help='List of files to compress. Multiple files and regex patterns are allowed.')
    parser_analyzer.set_defaults(which='analyzer')

    # Parse the command line arguments
    args = parser.parse_args()

    # Process options acording to the selected option
    if args.which is None:
        parser.print_help()
        exit(0)
    elif args.which == "compressor":
        from os.path import isdir, realpath
        from os import access, W_OK
        # Read the output folder from the command line and assert that it exists and has write permissions.
        output_folder = realpath(args.output_folder)
        assert isdir(output_folder), "The provided folder does not exist"
        assert access(output_folder, W_OK), "The output folder provided does not have write permissions"

        file_paths = args.files
        file_paths = sum(file_paths, [])

        # Compression options
        compression = args.compression
        # Read the number of nodes
        nodes = args.nodes

        # Import and launch compress function
        from enstools.io import compress
        compress(output_folder, file_paths, compression, nodes)
    elif args.which == "analyzer":
        file_paths = args.files
        # Compression options
        correlation = args.correlation
        # Output filename
        output_file = args.output
        from enstools.io import analyze
        analyze(file_paths, correlation, output_file)
