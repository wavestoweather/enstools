#!/usr/bin/env python
"""
    #
    # Prototype of the command line tool to compress datasets.
    #

"""


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



def analyze(file_paths, correlation_threshold=0.99999, output_file=None):
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
        
