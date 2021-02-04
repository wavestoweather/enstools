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
%prog /path/to/file/1 

-Multiple files:
%prog /path/to/file/1 /path/to/file/2

-Path pattern:
%prog /path/to/multiple/files/* 

- Specify tolerance
%prog /path/to/multiple/files/* --tolerance 0.999

"""
    parser = argparse.ArgumentParser(description=help_text,  formatter_class=argparse.RawDescriptionHelpFormatter)

    
    parser.add_argument("--correlation", dest="correlation", default=0.99999, type=float, 
                     help="Tolerance.")
    parser.add_argument("--output", "-o", dest="output", default=None, type=str)
    parser.add_argument("files", type=str, nargs="+")
    args = parser.parse_args()
    
    file_paths = args.files
    # Put all the files in a single 1d list
    #files = sum(file_paths,[])
    
       
    # Compression options
    correlation = float(args.correlation)
    
    # Output filename
    output_file = args.output
    # If we are not using MPI, just return the output folder and the list of files
    return file_paths, correlation, output_file


def zfp_analyze_variable(dataset,variable_name, mode, correlation_threshold = 0.99999):
    """
    Determine which ZFP parameters allow the reovered data to achieve the provided correlation threshold.
    Right now we are using the rate method and a simple iterative process in order to find the minimum rate value that still mantains the level of correlation.
    """
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
    """
    Load the dataset and go variable by variable determining the optimal ZFP compression parameters
    """
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
    """
    Function to switch between the different compressors and methods.
    Currrently only ZFP its available.
    """
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

        
def find_compression_parameters():
    """
    Finds optimal compression parameters for a list of files provided as command line arguments.
    The correlation_threshold can be adjusted by command line argument and if an output_file argument is provided it will output the json dictionary in there.
    """
    # Parse command line arguments
    file_paths, correlation_threshold, output_file = parse_command_line_arguments() 
    get_compression_parameters(file_paths,correlation_threshold, output_file)

if __name__ == "__main__":
    find_compression_parameters()
