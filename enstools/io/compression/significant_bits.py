"""
Functions to find which bits contain relevant information
using the approach described in Klöwer et al. 2021
(https://doi.org/10.1038/s43588-021-00156-2)

"""

import numpy as np
import xarray


def get_uint_type_by_bit_length(bits):
    """
    Returns the numpy data type that corresponds to a unsigned integer with a certain number of bits.
    """
    if bits == 8:
        return np.uint8
    elif bits == 16:
        return np.uint16
    elif bits == 32:
        return np.uint32
    elif bits == 64:
        return np.uint64
    else:
        raise NotImplementedError(f"mask_generator does not have this number of bits implemented: {bits}")


def mask_generator(bits=32, ones=9):
    """
    This function returns a mask of type unsigned int, which has the number of bits specified by the input variable bits 
    and has the number of ones specified by the input variable ones.

    Example:
        mask_generator(32,10) would return an unsigned integer whose binary
        representation would be 11111111110000000000000000000000

    FIXME: Right now we are not even considering little-big endian and these stuff.

    """
    mask_type = get_uint_type_by_bit_length(bits)
    # The approach to generate the mask is to generate a different single bit masks for each position that has a one
    # and then just merge them using logical_or
    elements = [single_bit_mask(position=i, bits=bits) for i in range(ones)]

    mask = mask_type(0.)
    for element in elements:
        mask = np.bitwise_or(mask, element)

    return mask_type(mask)


def single_bit_mask(position=0, bits=32):
    """
    It generates a mask that has a single 1 to the specified position.
    """
    # Get corresponding uint type
    mask_type = get_uint_type_by_bit_length(bits)
    # Get a 1 from the proper type
    mask = mask_type(1)
    # Shift 1 to the proper position
    shift = mask_type(bits - position - 1)
    shifted = np.left_shift(mask, shift, casting="unsafe")
    return mask_type(shifted)


def apply_mask(array: np.ndarray, mask):
    """
    Apply the mask to a given array.
    """
    # Get number of bits
    bits = array.itemsize * 8

    # Get the corresponding integer type
    int_type = get_uint_type_by_bit_length(bits)

    # Get a view of the array as the corresponding uint type.
    i_array = array.view(dtype=int_type)

    # Apply mask
    masked_array = np.bitwise_and(i_array, mask)

    # Return the masked array with the proper type.
    return masked_array.view(array.dtype)


def keep_significant_bits(array: np.ndarray, significant_bits=0, round_to_nearest=True):
    """
    Create and apply a mask with number of ones given by the input variable significant_bits.

    Rounding is controlled by round_to_nearest input variable.
    """

    bits = array.typesize * 8
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


def binary_representation(array: np.ndarray):
    """
    It returns a binary representation of an array.
    """
    bits = array.itemsize * 8
    array = array.ravel()

    int_type = get_uint_type_by_bit_length(bits)
    # Create a integer view of the data to be usable with binary_repr
    array = array.view(dtype=int_type)

    def wrapper(v):
        return np.binary_repr(v, width=bits)

    wrapper_vectorized = np.vectorize(wrapper)

    try:
        if isinstance(array, np.ndarray):
            return wrapper_vectorized(array)
        return np.binary_repr(array, bits)
    except Exception as err:
        print(type(array))
        raise err


def bit_in_position(array, position):
    assert array.ndim == 1

    # The number of bits per value its equal to the array type size in bytes multiplied by 8
    byte_length = array.dtype.itemsize
    bits = byte_length * 8

    shift = bits - position - 1
    mask = single_bit_mask(position=position, bits=bits)
    x = np.bitwise_and(array.view(dtype=mask.dtype), mask)
    x = np.right_shift(x, shift)

    return np.uint8(x)


def bit_count(array, position):
    x = bit_in_position(array, position=position)
    return np.sum(x)


def bit_probabilities(array):
    # The number of bits per value its equal to the array type size in bytes multiplied by 8
    byte_length = array.dtype.itemsize
    bit_length = byte_length * 8
    _bits = range(bit_length)

    counted_bits = [bit_count(array, pos) for pos in _bits]
    return [c / (len(array)) for c in counted_bits]


def entropy_(p):
    s = 0.
    z = 0.
    for pi in p:
        if pi > z:
            s += pi * np.log(pi)
    return -s if s < 0 else s


def entropy__(p, base):
    return entropy_(p) / np.log(base)


def bitpattern_entropy_per_bit(array):
    probabilities = bit_probabilities(array)
    return [entropy__([p, 1 - p], base=2) for p in probabilities]


def bitpattern_entropy(A):
    return new_entropy(A)


def new_entropy(array):
    unique, counts = np.unique(array, return_counts=True)
    probabilities = counts / array.size
    p = probabilities
    E = - np.sum(p * np.log2(p))
    return E


def bit_conditional_count(array):
    l_not = np.logical_not

    as_bool = array.astype(np.bool)
    rolled = np.roll(as_bool, 1)

    zero_zero = np.logical_and(l_not(as_bool), l_not(rolled))
    zero_one = np.logical_and(l_not(as_bool), rolled)
    one_zero = np.logical_and(as_bool, l_not(rolled))
    one_one = np.logical_and(as_bool, rolled)

    counter = np.zeros(shape=(2, 2))
    counter[0][0] = np.sum(zero_zero)
    counter[0][1] = np.sum(zero_one)
    counter[1][0] = np.sum(one_zero)
    counter[1][1] = np.sum(one_one)

    return counter


def bit_mutual_information(array):
    counter = bit_conditional_count(array)
    probabilities = counter / array.size
    p = probabilities

    p1 = np.sum(array) / array.size
    p0 = 1. - p1

    pc = np.zeros(shape=(2, 2))
    pc[0][0] = (p[0][0] / p0) if p0 > 0. else 0
    pc[0][1] = (p[0][1] / p0) if p0 > 0. else 0
    pc[1][0] = (p[1][0] / p1) if p1 > 0. else 0
    pc[1][1] = (p[1][1] / p1) if p1 > 0. else 0

    H0 = - ((pc[0][0] * np.log2(pc[0][0]) if pc[0][0] > 0 else 0) + (
        pc[0][1] * np.log2(pc[0][1]) if pc[0][1] > 0 else 0))
    H1 = - ((pc[1][0] * np.log2(pc[1][0]) if pc[1][0] > 0 else 0) + (
        pc[1][1] * np.log2(pc[1][1]) if pc[1][1] > 0 else 0))

    H = entropy__([p0, p1], 2)

    mutual_information = H - p0 * H0 - p1 * H1
    if mutual_information < 0:
        mutual_information = 0
    return mutual_information


def array_mutual_information(array):
    """
    Given an array, look for the mutual information in each bit position.
    Returns a list of the mutual information in each bit.
    """
    bit_length = array.itemsize * 8

    mutual_information = np.zeros(shape=bit_length, dtype=np.float32)
    for pos in range(bit_length):
        bits = bit_in_position(array, position=pos)
        mutual_information[pos] = bit_mutual_information(bits)

    return mutual_information


def analyze_file_significant_bits(file_path):
    # analyze(file_paths, output_file, thresholds, compressor=compressor, mode=mode, grid=grid)
    from enstools.io import read

    # Open file
    dataset = read(file_path)

    # Get list of variables
    variables = [var for var in dataset.variables if var not in dataset.coords]

    # Analyze variable per variable
    for variable in variables:
        # if variable != "tot_prec": continue
        analyze_variable_significant_bits(dataset[variable])


def analyze_variable_significant_bits(data_array: xarray.DataArray):
    data_array = data_array.squeeze()

    var_name = data_array.name
    print(f"Variable:{var_name}")
    assert data_array.ndim >= 3

    # We will compute the mutual information on the first frame and the last frame
    # in order to know if there's important variability
    first_frame = data_array[0]
    last_frame = data_array[-1]

    # Get data
    first_frame_data = first_frame.values.ravel()
    last_frame_data = last_frame.values.ravel()

    # Apply fix
    first_frame_data = fix_repetition(first_frame_data)
    last_frame_data = fix_repetition(last_frame_data)

    # Compute mutual information
    first_frame_mi = array_mutual_information(first_frame_data)
    last_frame_mi = array_mutual_information(last_frame_data)

    # Sum all information
    first_frame_total_information = np.sum(first_frame_mi)
    last_frame_total_information = np.sum(last_frame_mi)

    # There can be small variations but we want to know if there's something big happening.
    if np.abs(first_frame_total_information - last_frame_total_information) > 0.5:
        print("Warning, first frame is unreliable, checking all data")
        max_info = max(first_frame_total_information, last_frame_total_information)
        max_mutual_info = first_frame_mi
        for frame in data_array:
            frame_data = frame.values.ravel()
            frame_data = fix_repetition(frame_data)
            tmp_mi = array_mutual_information(frame_data)
            print(f"{sum(tmp_mi):.2f}")
            mutual_information_report(tmp_mi, var_name)
            if sum(tmp_mi) >= max_info:
                max_info = sum(tmp_mi)
                max_mutual_info = tmp_mi
        first_frame_mi = max_mutual_info
    mi = first_frame_mi
    mutual_information_report(mi, var_name)


def mutual_information_report(mutual_information_list, name=""):
    # import matplotlib.pyplot as plt
    bits = len(mutual_information_list)

    if bits == 32:
        first_mantissa_bit = 10
    elif bits == 64:
        first_mantissa_bit = 12

    exponent = slice(1, first_mantissa_bit)
    mantissa = slice(first_mantissa_bit, bits + 1)

    sign_bit = mutual_information_list[0]

    exponent_bits = mutual_information_list[exponent]
    mantissa_bits = mutual_information_list[mantissa]

    # FIXME: Get rid of insignificant values
    # In the code created by the author of the paper they use this:
    """
    if set_zero_insignificant
        p = binom_confidence(N,confidence)  # get chance p for 1 (or 0) from binom distr
        I₀ = 1 - entropy([p,1-p],2)         # free entropy of random [bit]
        I[I .<= I₀] .= 0                    # set insignficant to zero
    end
    """

    print(f"Exponent information:{sum(exponent_bits):.1f}")
    print(f"Mantissa information:{sum(mantissa_bits):.1f}")

    mantissa_info = sum(mantissa_bits)

    preserved_information = .99

    cumulated = 0.0
    index = 0
    while cumulated < mantissa_info * preserved_information:
        try:
            cumulated += mantissa_bits[index]
            index += 1
        except IndexError:
            break
    print(f"NSB:{index}")
    # plt.bar(range(len(mantissa_bits)), mantissa_bits)
    # plt.savefig(f"mantissa_information_{name}.png")
    # plt.clf()


def fix_repetition(array: np.array) -> np.array:
    """
    This function gets a numpy array and returns the same array without consecutive values repeated.
    Parameters
    ----------
    array

    Returns
    -------
    array
    """
    shifted = np.roll(array, 1)
    indices = array != shifted
    return array[indices]
