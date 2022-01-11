from genericpath import isfile
import xarray
import numpy as np
from enstools.io.compression import significant_bits
from enstools.io.compression.significant_bits import get_uint_type_by_bit_length, single_bit_mask, apply_mask, mask_generator
from enstools.io import read, write

def prune_numpy_array(array: np.array, significant_bits=0, round_to_nearest=True):
    """
    Create and apply a mask with number of ones given by the input variable significant_bits.

    Rounding is controlled by round_to_nearest input variable.
    """

    bits = array.dtype.itemsize * 8
    int_type = get_uint_type_by_bit_length(bits)

    # round_to_nearest = True
    if round_to_nearest:
        # Get the mask for the first discarded value
        next_bit_mask = single_bit_mask(position=significant_bits, bits=bits)

        # Apply the mask
        next_bit_value = apply_mask(array, next_bit_mask)

        # Shift left
        next_bit_value = np.left_shift(next_bit_value.view(dtype=int_type), 1)

        # Apply or
        new_array = np.bitwise_or(array.view(dtype=int_type), next_bit_value).view(dtype=array.dtype)

        # Reasign
        array = new_array[:]

    # Create mask
    mask = mask_generator(bits=bits, ones=significant_bits)
    # Apply mask
    pruned = apply_mask(array, mask)
    assert len(pruned) == len(array)

    return pruned

def pruner(file_paths, output, significant_bit_info=None):
    from enstools.io.compression.compressor import destination_path
    from os.path import isdir
    from os import rename, access, W_OK

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
        prune_file(file_path, new_file_path, significant_bit_info=significant_bit_info)
    elif len(file_paths) > 1:
        # In case of having more than one file, check that output corresponds to a directory
        assert isdir(output), "For multiple files, the output parameter should be a directory"
        assert access(output, W_OK), "The output folder provided does not have write permissions"

        

def prune_file(file_path, destination, significant_bit_info=None):
    print(f"{file_path} -> {destination}")
    ds = read(file_path)
    if significant_bit_info is None:
        from enstools.io.compression.significant_bits import analyze_file_significant_bits
        significant_bits_dictionary = analyze_file_significant_bits(file_path)
    elif isfile(significant_bit_info):
        
        significant_bits_dictionary = {}
    prune_dataset(ds, significant_bits_dictionary)

    # After pruning the file we save it to the new destination.
    write(ds, destination, compression="lossless")

def prune_dataset(dataset: xarray.Dataset, significant_bits_dictionary: dict):
    variables = [ v for v in dataset.variables if v not in dataset.coords]

    for variable in variables:
        try:
            significant_bits = significant_bits_dictionary[variable]
        except KeyError:
            print(f"Warning: number of significant bits for variable {variable} has not been provided.")
            continue
        prune_data_array(dataset[variable], significant_bits)

def prune_data_array(data_array: xarray.DataArray, significant_bits: int):
    data_array.values = prune_numpy_array(data_array.values, significant_bits=significant_bits)
