#!/usr/bin/env python
"""
    #
    # Prototype of the command line tool to compress datasets.
    #

"""
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
%prog /path/to/file/1 

-Multiple files:
%prog /path/to/file/1 /path/to/file/2

-Path pattern:
%prog /path/to/multiple/files/* 

- Specify tolerance
%prog /path/to/multiple/files/* --tolerance 0.999

"""
    
    parser = OptionParser(usage=help_text)
    
    parser.add_option("--tolerance", dest="tolerance", default="0.99999",
                     help="Tolerance.")
    parser.add_option("--output", "-o", dest="output", default=None)

    (options, args) = parser.parse_args()
    if len(args) == 0:
        print("No argument given!")
        parser.print_help()
        exit(1)
    
    # Expand the filenames in case we ued a regex expression
    assert len(args) > 0, "Need to provide the list of files as arguments. Regex patterns are allowed."
    file_paths = sum([glob.glob(arg) for arg in args], [])

    # Compression options
    tolerance = float(options.tolerance)
    
    # Output filename
    output_file = options.output
    # If we are not using MPI, just return the output folder and the list of files
    return file_paths, tolerance, output_file


def zfp_analyze_variable(dataset,variable_name, mode, correlation_threshold = 0.99999):
    import zfpy
    import numpy as np
    from scipy.stats.stats import pearsonr
    try:
        variable_data = dataset[variable_name].sel(time=dataset["time"][0])
    except IndexError:
        variable_data = dataset[variable_name]
    
    variable_data = np.squeeze(variable_data.values)
    
    rate = 0
    corr = 0
    while corr < correlation_threshold:
        rate += 1
        compressed_data = zfpy.compress_numpy(variable_data,rate=rate)
        recovered_data = zfpy.decompress_numpy(compressed_data)
        corr, pval = pearsonr(variable_data.ravel(), recovered_data.ravel())
        if np.isnan(corr):
            corr = 0
    return "lossy:zfp:rate:%i" %rate


def zfp_analyze_files(file_paths, correlation_threshold=0.99999):
    from .reader import read
    dataset = read(file_paths)
    variables = [v for v in dataset.variables]
    
    encoding = {}
    for v in variables:
        if dataset[v].size < 10000:
            encoding[v] = "lossless"
        else:
            encoding[v] = zfp_analyze_variable(dataset, v, mode="rate", correlation_threshold=correlation_threshold)
    return encoding


def analyze_files(file_paths, correlation_threshold=0.99999):
    return zfp_analyze_files(file_paths, correlation_threshold)



def get_compression_parameters(file_paths,correlation_threshold=0.99999, output_file=None):
    """
    Finds optimal compression parameters for a list of files to fullfill a correlation_threshold. 
    If an output_file argument is provided it will output the json dictionary in there.
    """

    print("Analyzing files to determine optimal compression options to achieve a Pearson Correlation of %f ." % correlation_threshold)
    encoding = analyze_files(file_paths, correlation_threshold)
    
    import json
    if output_file:
        print("Compresion options saved in: %s " % output_file)
        with open(output_file, "w") as outfile:
            json.dump(encoding, outfile,indent=4, sort_keys=True)
    else:
        print("Compresion options:")
        print(json.dumps(encoding,indent=4, sort_keys=True))

        
def main():
    """
    Finds optimal compression parameters for a list of files provided as command line arguments.
    The correlation_threshold can be adjusted by command line argument and if an output_file argument is provided it will output the json dictionary in there.
    """
    # Parse command line arguments
    file_paths, correlation_threshold, output_file = parse_command_line_arguments() 
    get_compression_parameters(file_paths,correlation_threshold, output_file)

if __name__ == "__main__":
    main()
