#!/usr/bin/env python
"""
    #
    # Routines to find optimal compression parameters to satisfy certain quality thresholds
    #

"""

from dask.config import get
from .metrics import DataArrayMetrics
from pprint import pprint
import numpy as np
from xarray import DataArray, Dataset
from typing import Union, List, Tuple
from .encoding import check_libpressio_availability
from sys import getsizeof

compression_modes = {
    "zfp": ["accuracy",
            "rate",
            "precision",
            ],
    "sz": ["abs",
            "rel",
            "pw_rel",
            ]
}

def check_thresholds(recovered_data: np.ndarray, reference_data: np.ndarray, thresholds:dict):
    """
    Function to check if all the defined thresholds are fulfilled
    """
    if recovered_data is None:
        return False
    metrics = DataArrayMetrics(reference_data, recovered_data)
    for method, target in thresholds.items():
        result = metrics[method]
        # Depending on the method, the threshold its a lower or an upper bound
        if method in ["correlation_I", "ssim_I", "nrmse_I"]:
            if result < target:
                return False
        else:
            if result > target:
                return False
    return True


def compressor_configuration(compressor:str, mode:str, parameter:float, dataset:Union[np.ndarray, DataArray]):
    if compressor == "sz":
        compressor_config =  {
                "sz:error_bound_mode_str": mode,
                "sz:abs_err_bound": parameter,
                "sz:rel_err_bound": parameter,
                "sz:pw_rel_err_bound":parameter,
                "sz:metric": "size",
                }
    elif compressor == "zfp":
        compressor_config = {
        f"zfp:{mode}": parameter,
        "zfp:type": 1,
        "zfp:dims": len(dataset.shape),
        "zfp:wra": 1,
        "zfp:execution_name": mode,
        "zfp:metric": "size",
        }
    else:
        raise NotImplementedError( f"{compressor} {mode}")
    return {
        "compressor_id": compressor,
        "compressor_config": compressor_config,
        }


def analyze_variable(dataset: DataArray, variable_name:str, thresholds:dict, compressor_name:str, mode:str) -> Tuple[str, float]:
    import numpy as np
    import logging
    try:
        from libpressio import PressioCompressor
    except ModuleNotFoundError as err:
        print("The library libpressio its not available, can not proceed with analysis.")
        raise err

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
        logging.info("The data of the following variable contains NaN: %s, falling back to BLOSC compression." % variable_name)
        return "lossless"
    if len(variable_data.shape) == 1:
        logging.info("1D variable found: %s, falling back to BLOSC compression." % variable_name)
        return "lossless"
    

    # Set buffers
    uncompressed_data = variable_data
    decompressed_data = uncompressed_data.copy()
    recovered_data = None

    parameter, step, exit_condition = search_parameters(compressor_name, mode, variable_data)
    while not check_thresholds(recovered_data, variable_data, thresholds):
        parameter = step(parameter)
        if exit_condition(parameter):
            return "lossless", 1.0
        config = compressor_configuration(compressor_name, mode, parameter, variable_data)
        compressor = PressioCompressor.from_config(config)
        
        # preform compression and decompression
        compressed = compressor.encode(uncompressed_data)
        decompressed = compressor.decode(compressed, decompressed_data)
        recovered_data = decompressed
    metrics = compressor.get_metrics()
    compression_ratio = metrics['size:compression_ratio']
    compression_spec = f"lossy:{compressor_name}:{mode}:{parameter}"
    return compression_spec, compression_ratio


def zfp_analyze_variable(dataset: DataArray, variable_name:str, compressor_name:str, mode:str, thresholds:dict):
    """
    Ideally it won't be necessary, only to be used when libpressio its not available and zfpy is.
    Determine which ZFP parameters allow the recovered data to achieve the provided correlation threshold.
    Right now we are using the rate method and a simple iterative process in order to find the minimum rate value that
    still maintains the level of correlation.
    """
    # Argument compressor_name was kept only to mantain the same api that the generic analyze_variable function.
    # We should not enter this function with a compressor name different than zfp
    assert compressor_name == "zfp", "Trying to use zfp_analyze_variable with a different compressor than zfp."

    # There's a mismatch between how tolerance mode is called in zfp.
    # In zfpy it is called tolerance, in libpressio it is called accuracy.
    if mode == "accuracy":
        mode = "tolerance"

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
    
    original_size = variable_data.size * variable_data.itemsize
    #print(f"Original size: {original_size}")
    recovered_data=None
    parameter, step, exit_condition = search_parameters(compressor_name, mode, variable_data)
    while not check_thresholds(recovered_data, variable_data, thresholds):
        parameter = step(parameter)
        if exit_condition(parameter):
            return "lossless", 1.0
        compressor_parameters = {mode:parameter}
        compressed_data = zfpy.compress_numpy(variable_data, **compressor_parameters)
        
        compressed_size = getsizeof(compressed_data)
        compression_ratio = original_size/compressed_size
        #print(f"Compressed size: {compressed_size} CR:{compression_ratio:.1f}")

        recovered_data = zfpy.decompress_numpy(compressed_data)
    if mode == "tolerance":
        mode = "accuracy"
    return f"lossy:zfp:{mode}:{parameter}", compression_ratio



def search_parameters(compressor:str, mode:str, data_values:np.ndarray):
    """
    Depending on the compressor and the mode used, the search its performed in one way or another.
    I.e. in accuracy (or abs) mode, we'll start with a big value and will decrease it until the thresholds are fullfilled,
    in rate mode, we'll follow the opposite approach, we start with a small value and increase it until the thresholds are fullfilled.
    This function defines the starting value, the operation that we perform at each step and the exit condition.
    """
    if compressor == "zfp":
        if mode == "rate":
            parameter = 2 # Starting value
            step = lambda x: x+.25
            def exit_condition(_parameter):
                return _parameter > 16
        elif mode == "precision":
            parameter = 8.
            step = lambda x: x+1.
            def exit_condition(_parameter):
                return _parameter > 64
        elif mode == "accuracy" or mode == "tolerance":
            from math import log10, floor
            def round_to_1(x):
                return round(x, -int(floor(log10(abs(x)))))
            parameter = round_to_1((np.max(data_values) - np.min(data_values)) / 1.)
            p0 = parameter
            factor = 2
            step = lambda x: x/factor

            def exit_condition(_parameter):
                return 10**-6 * p0 > _parameter
        
        else:
            raise NotImplementedError( f"{compressor} {mode}")
        return parameter, step, exit_condition
    elif compressor == "sz":
        if mode == "pw_rel" or mode == "rel":
            parameter = 1e-0  # The process will start at rate 3
            step = lambda x: x/2.
            def exit_condition(parameter):
                return parameter < 1e-10
        elif mode == "abs":
            from math import log10, floor
            def round_to_1(x):
                return round(x, -int(floor(log10(abs(x)))))
            parameter = round_to_1((np.max(data_values) - np.min(data_values)) / 100.)
            p0 = parameter
            factor = 2
            step = lambda x: x/factor

            def exit_condition(parameter):
                return 10**-6 * p0 > parameter
        else:
            raise NotImplementedError( f"{compressor} {mode}")
        return parameter, step, exit_condition


def analyze_files(file_paths:List[str], thresholds:dict ,compressor:str="zfp",mode:str=None):
    """
    Load the dataset and go variable by variable determining the optimal compression parameters
    """

    # Check library availabilty and select proper analysis_function
    if check_libpressio_availability():
        analysis_function = analyze_variable
        compressors = ["zfp", "sz"]
    else:
        analysis_function = zfp_analyze_variable
        compressors = ["zfp"]

    if mode is None:
        multimode = True # Try between the different compressor methods to select the better performing one.
    else:
        multimode = False
    from .reader import read
    dataset = read(file_paths)
    variables = [v for v in dataset.variables]

    encoding = {}
    for var in variables:
        if dataset[var].size < 10000:
            encoding[var] = "lossless"
        else:
            if multimode:
                highest_compression_ratio = 0.
                selected_encoding = None
                for compressor in compressors:
                    for mode in compression_modes[compressor]:
                        variable_encoding, compression_ratio = analysis_function( dataset,
                                                    variable_name=var,
                                                    compressor_name=compressor,
                                                    mode=mode,
                                                    thresholds=thresholds
                                                    )
                        print(f"{var} {mode} CR:{compression_ratio:.1f}")
                        if compression_ratio > highest_compression_ratio:
                            selected_encoding = variable_encoding
                            highest_compression_ratio = compression_ratio
                print(f"{var} Selected->{selected_encoding}  CR:{highest_compression_ratio:.1f}")
                encoding[var] = selected_encoding
            else:
                variable_encoding, compression_ratio = analysis_function( dataset,
                                                variable_name=var,
                                                compressor_name=compressor,
                                                mode=mode,
                                                thresholds=thresholds
                                                )
                encoding[var] = variable_encoding
            #(dataset, variable_name, thresholds, compressor_name, mode)
    return encoding


def analyze(file_paths:List[str], output_file:str=None,  thresholds:dict=None,format:str="yaml", compressor:str="zfp", mode:str=None):
    """
    Finds optimal compression parameters for a list of files to fulfill certain thresholds.
    If an output_file argument is provided it will output the dictionary in there (yaml or json allowed).
    """

    if thresholds is None:
                thresholds = {
                        "correlation_I": 5,
                        "ssim_I": 4,
                       }

    print(f"\nAnalyzing files to determine optimal compression options for compressor {compressor} with mode {mode} to fullfill the following thresholds:")
    pprint(thresholds)
    print()
    encoding = analyze_files(file_paths, thresholds, compressor=compressor, mode=mode)

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
