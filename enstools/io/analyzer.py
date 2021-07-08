#!/usr/bin/env python
"""
    #
    # Prototype of the command line tool to compress datasets.
    #

"""


def zfp_analyze_variable(dataset, variable_name, mode, correlation_threshold=0.99999):
    """
    Determine which ZFP parameters allow the recovered data to achieve the provided correlation threshold.
    Right now we are using the rate method and a simple iterative process in order to find the minimum rate value that
    still maintains the level of correlation.
    """
    try:
        import zfpy
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Module zfpy not found")
    import numpy as np
    from scipy.stats.stats import pearsonr
    import logging
    try:
        variable_data = dataset[variable_name].sel(time=dataset["time"][0])
    except IndexError:
        variable_data = dataset[variable_name]
    except ValueError:
        variable_data = dataset[variable_name]
    variable_data = np.squeeze(variable_data.values)
    # Check if the array contains any nan
    contains_nan = np.isnan(variable_data).any()

    if contains_nan:
        logging.debug("The data of the following variable contains NaN: %s, falling back to BLOSC compression." % variable_name)
        return "lossless"
    # Replace NaNs with ones
    #variable_data[np.isnan(variable_data)] = 1
    if len(variable_data.shape) == 1:
        logging.debug("1D variable found: %s, falling back to BLOSC compression." % variable_name)
        return "lossless"
    rate = 2  # The process will start at rate 3
    corr = 0
    while corr < correlation_threshold:
        rate += 1
        if rate >= 12:
            # In case of requiring a rate bigger than 12 we might just jump to lossless compression for simplicity.
            return "lossless"
        logging.debug("Analysis-> var: %s Trying rate: %i" % (variable_name, rate))
        compressed_data = zfpy.compress_numpy(variable_data, rate=rate)
        recovered_data = zfpy.decompress_numpy(compressed_data)
        corr, pval = pearsonr(variable_data.ravel(), recovered_data.ravel())
        if np.isnan(corr):
            corr = 0
    return "lossy:zfp:rate:%i" % rate


def sz_analyze_variable(dataset, variable_name, mode, correlation_threshold=0.99999):
    """
    Determine which ZFP parameters allow the recovered data to achieve the provided correlation threshold.
    Right now we are using the rate method and a simple iterative process in order to find the minimum rate value that
    still maintains the level of correlation.
    """
    try:
        import pysz
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Python module for SZ not found")
    import numpy as np
    from scipy.stats.stats import pearsonr
    import logging
    try:
        variable_data = dataset[variable_name].sel(time=dataset["time"][0])
    except IndexError:
        variable_data = dataset[variable_name]
    except ValueError:
        variable_data = dataset[variable_name]

    variable_data = np.squeeze(variable_data.values)

    if len(variable_data.shape) == 1:
        logging.debug("1D variable found: %s, falling back to BLOSC compression." % variable_name)
        return "lossless"
    rel = 1  # The process will start at rel 1
    corr = 0
    while corr < correlation_threshold:
        rel *= .1
        if rel <= 10**-5:
            # In case of requiring a rate bigger than 12 we might just jump to lossless compression for simplicity.
            return "lossless"
        logging.debug("Analysis-> var: %s Trying pw_rel: %f" % (variable_name, rel))
        print("Original:", variable_data.sum())

        compressor = pysz.Compressor((pysz.ConfigBuilder().errorBoundMode(pysz.PW_REL)
                                      .pw_relBoundRatio(rel).build()))
        compressed_data = compressor.Compress(variable_data)
        recovered_data = compressor.Decompress(compressed_data, variable_data.shape, np.float32)
        print("Recovered:", recovered_data.sum())
        del compressor

        corr, pval = pearsonr(variable_data.ravel(), recovered_data.ravel())
        print(corr)
        if np.isnan(corr):
            corr = 0
    return "lossy:sz:pw_rel:%f" % rel


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


def sz_analyze_files(file_paths, correlation_threshold=0.99999):
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
            encoding[v] = sz_analyze_variable(dataset, v, mode="rel", correlation_threshold=correlation_threshold)
    return encoding


def analyze_files(file_paths, correlation_threshold=0.99999, compressor="zfp"):
    """
    Function to switch between the different compressors and methods.
    Currrently only ZFP its available.
    """
    if compressor == "sz":
        return sz_analyze_files(file_paths, correlation_threshold)
    elif compressor == "zfp":
        return zfp_analyze_files(file_paths, correlation_threshold)
    else:
        raise NotImplementedError(f"This compressor has not been implemented yet: {compressor}")


def analyze(file_paths, correlation_threshold=0.99999, output_file=None,format="yaml"):
    """
    Finds optimal compression parameters for a list of files to fulfill a correlation_threshold.
    If an output_file argument is provided it will output the json dictionary in there.
    """

    print(
        "Analyzing files to determine optimal compression options to achieve a Pearson Correlation of %f ."
        % correlation_threshold)
    encoding = analyze_files(file_paths, correlation_threshold)

    if format == "json":
        
        import json

        if output_file:
            print("Compression options saved in: %s " % output_file)
            with open(output_file, "w") as outfile:
                json.dump(encoding, outfile, indent=4, sort_keys=True)
        else:
            print("Compression options:")
            print(json.dumps(encoding, indent=4, sort_keys=True))
    elif format == "yaml":
        import yaml
        if output_file:
            print("Compression options saved in: %s " % output_file)
            with open(output_file, "w") as outfile:
                yaml.dump(encoding, outfile, sort_keys=True)
        else:
            print("Compression options:")
            print(yaml.dump(encoding, indent=4, sort_keys=True))
