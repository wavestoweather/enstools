from typing import List, Union, Tuple

import xarray

from enstools.io.compression.compressors.zfp import zfp_encoding, parse_zfp_compression_options
from enstools.io.compression.compressors.sz import sz_encoding, parse_sz_compression_options
from enstools.io.compression.compressors.blosc import blosc_encoding, parse_blosc_compression_options

from os.path import isfile

compressor_labels = {
    32001: "BLOSC",
    32013: "ZFP",
    32017: "SZ",
}


def define_encoding(dataset: xarray.Dataset, compression: Union[str, None]) -> tuple:
    """
    Create a dictionary with the encoding that will be passed to the hdf5 engine.

    Parameters
    ----------
    dataset : xarray.Dataset
            the dataset that will be stored

    compression: string or filepath
        It can contain the string that defines the compression options that will be used in the whole dataset,
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
        For lossy compression, we can choose the compressor (zfp or sz),
        the method and the method parameter (the accuracy, the precision or the rate).
        Some examples:
            "lossless"
            "lossy"
            "lossless:zlib:5"
            "lossy:zfp:accuracy:0.00001"
            "lossy:zfp:precision:12"
            "lossy:sz:abs:0.1"
            "lossy:sz:rel:0.01"

        If using a configuration file, the file should follow a json format and can contain per-variable values.
        It is also possible to define a default option. For example:
        { "default": "lossless",
          "temp": "lossy:zfp:accuracy:0.1",
          "qv": "lossy:zfp:accuracy:0.00001"
        }



    """
    if compression is None or compression == "None":
        return None, None
    # First we create a dictionary with the compression specifications
    specifications_per_variable = define_variable_specifications(dataset, compression=compression)

    # Second we create a dictionary with the arguments that we need to pass to xarray.
    compression_options_per_variable = define_compression_options(dataset, specifications_per_variable)

    # Create descriptions to be used for the variable attributes
    descriptions = create_descriptions(variable_specifications=specifications_per_variable,
                                       encoding=compression_options_per_variable)
    return compression_options_per_variable, descriptions


def define_variable_specifications(dataset: xarray.Dataset, compression: Union[str, dict]) -> dict:
    """
    This function will return a dictionary with a string of compression specifications for each variable.
    The format of these specifications will be a string following our syntax : i.e. lossy:zfp:rate:4.0 , lossless , ...
    Parameters
    ----------
    dataset
    compression: can be a dictionary with a specification for some or all the variables, a string containing
                 the specification or a string containing the path to a configuration file.

    Returns
    -------

    """
    # Possible cases:
    # dictionary: we have specification per variable directly from a dictionary
    # filename: we have specification per variable from a configuration file
    # lossless: same specification for all variables
    # lossy: lossless specification for coordinates and lossy compression for data variables
    if compression is None or compression == "None":
        return define_variable_specifications_none(dataset)
    elif isinstance(compression, dict):
        return define_variable_specification_from_dictionary(dataset, compression)
    elif isfile(compression):
        return define_variable_specification_from_filename(dataset, compression)
    elif compression.count("lossless"):
        return define_variable_specifications_lossless(dataset, compression)
    elif compression.count("lossy"):
        return define_variable_specifications_lossy(dataset, compression)
    elif compression == "default":
        return define_variable_specifications_none(dataset)
    else:
        raise NotImplementedError


def define_variable_specification_from_dictionary(dataset: xarray.Dataset, compression: dict) -> dict:
    specifications = {}
    coordinates, data_variables = coordinates_and_variables_from_dataset(dataset)

    try:
        default = compression["default"]
    except KeyError:
        default = "lossless"

    # Go through coordinates:
    for variable in coordinates + data_variables:
        try:
            specifications[variable] = compression[variable]
        except KeyError:
            specifications[variable] = default

    return specifications


def define_variable_specification_from_filename(dataset: xarray.Dataset, compression: str) -> dict:
    specification_dictionary = parse_compression_configuration_file(compression)
    return define_variable_specification_from_dictionary(dataset, specification_dictionary)


def define_variable_specifications_lossless(dataset: xarray.Dataset, compression: str) -> dict:
    coordinates, data_variables = coordinates_and_variables_from_dataset(dataset)
    definitions = {}
    # Go through all variables:
    for variable in coordinates + data_variables:
        definitions[variable] = compression
    return definitions


def define_variable_specifications_none(dataset: xarray.Dataset) -> dict:
    coordinates, data_variables = coordinates_and_variables_from_dataset(dataset)
    definitions = {}
    # Go through all variables:
    for variable in coordinates + data_variables:
        definitions[variable] = None
    return definitions


def define_variable_specifications_lossy(dataset: xarray.Dataset, compression: str) -> dict:
    coordinates, data_variables = coordinates_and_variables_from_dataset(dataset)
    definitions = {}
    # Go through coordinates:
    for variable in coordinates:
        definitions[variable] = "lossless"

    for variable in data_variables:
        definitions[variable] = compression
    return definitions


def coordinates_and_variables_from_dataset(dataset: xarray.Dataset) -> Tuple[list, list]:
    """
    Given a dataset, it returns a list of coordinate names and a list of data variable names
    Parameters
    ----------
    dataset

    Returns
    -------
    coordinates
    non_coordinates
    """
    coordinates = [v for v in dataset.coords]
    non_coordinates = [v for v in dataset.variables if v not in coordinates]

    return coordinates, non_coordinates


def define_compression_options(dataset: xarray.Dataset, variable_specifications: dict):
    compression_options = {}
    for variable, specification in variable_specifications.items():
        mode, options = parse_compression_options(specification)
        if mode is None:
            filter_id = parsed_options = None
        elif mode == "lossless":
            compressor, compression_level = options
            filter_id, parsed_options = lossless_encoding(compressor=compressor, compressor_level=compression_level)
        elif mode == "lossy":
            filter_id, parsed_options = lossy_encoding(options)

            parsed_options = adapt_compression_options(filter_id=filter_id,
                                                       compression_options=parsed_options,
                                                       variable_dataset=dataset[variable],
                                                       )
        else:
            raise NotImplementedError

        # Fix to get chunksizes
        chunksize = tuple([_x if i != 0 else 1 for i, _x in enumerate(dataset[variable].shape)])

        compression_options[variable] = {
            "compression": filter_id,
            "compression_opts": parsed_options,
            "chunksizes": chunksize,
        }
    return compression_options


def create_descriptions(variable_specifications, encoding) -> dict:
    descriptions = {}
    for variable, specification in variable_specifications.items():
        # In case of not using any compression, we will just avoid writing the attribute.
        if specification is None:
            continue
        filter_id = encoding[variable]["compression"]
        descriptions[variable] = f"Compressed with Ensemble Tools using: {specification} " \
                                 f"(Using {compressor_labels[filter_id]} " \
                                 f"filter with id: {filter_id})"

    return descriptions


def set_compression_attributes(dataset: xarray.Dataset, descriptions: Union[dict, None]):
    # In case of using an encoding, we'll add an attribute to the file indicating that the file has been compressed.
    if descriptions:
        for variable, description in descriptions.items():
            if description is not None:
                dataset[variable].attrs["compression"] = description


def adapt_compression_options(filter_id: int, compression_options: tuple, variable_dataset: xarray.DataArray) -> tuple:
    """
    In the case of SZ (filter_id == 32017) we need to adapt the compression options to include things related with
    the dimensions of the variable being dealt with.
    :param filter_id:
    :param compression_options:
    :param variable_dataset:
    :return:
    """
    if filter_id == 32013:  # Case ZFP
        return compression_options
    elif filter_id == 32017:  # Case SZ
        # Case SZ (filter id 32017)
        # Compression_opts have to include dimensions of data and shape.
        shape = variable_dataset.shape
        dim = len(shape)
        # TODO: We have to set the type dynamically instead of having it hardcoded here. Also for integers.
        data_type = 0

        if variable_dataset.dtype == "float32":
            data_type = 0
        elif variable_dataset.dtype == "float64":
            data_type = 1
        error_mode = compression_options[0]
        error_val = compression_options[2]
        chunksize = tuple([_x if i != 0 else 1 for i, _x in enumerate(variable_dataset.shape)])

        return (dim, data_type, *chunksize, error_mode, error_val, error_val, error_val, error_val)
    else:  # Other cases (Blosc?)
        return compression_options


def lossy_encoding(compression_options: List[str]) -> Union[int, tuple]:
    """
     Returns the filter id and a tuple with the encoding that will be passed to the hdf5 engine.
     Its a wrapper that will select the proper function to use: zfp_encoding or sz_encoding

    Parameters
    ----------
    compression_options: list
        list of compression options
    """

    if compression_options[0] == "zfp":
        return zfp_encoding(compression_options)
    elif compression_options[0] == "sz":
        return sz_encoding(compression_options)
    else:
        raise NotImplementedError


def lossless_encoding(compressor: str, compressor_level: int) -> Union[int, tuple]:
    return blosc_encoding(compressor=compressor, compressor_level=compressor_level)


# Some functions to define the compression_opts array that will be passed to the filter
#############################################
# Functions to parse the compression options,
# to give more flexibility and allow a way to pass custom compression options
# TODO: Move to a different file?
def parse_compression_options(string: str) -> tuple:
    """
    Function to parse compression options
    Input
    -----
    string: string
    
    Output
    -----
    compression_method : string
                        "lossy", "lossless" or None
    compression_opts:  tuple
                       for lossless: (backend, clevel) 
                       for lossy:    (backend, method, parameter)
    """
    # In case of the argument being None or "None" just return None, None
    if string is None or string == "None":
        return None, None
    arguments = string.split(":")
    # Check that all arguments have information
    assert not any(
        [not argument.strip() for argument in arguments]), "Compression: The provided option has a wrong format."
    first = arguments[0]
    # Check if it is a file:
    if first == "lossless":
        return parse_lossless_compression_options(arguments)
    elif first == "lossy":
        return parse_lossy_compression_options(arguments)
    else:
        raise AssertionError("Compression: The argument should be None/lossless/lossy.")


def parse_lossless_compression_options(arguments: List[str]):
    """
    Function to parse compression options for the lossless case
    Input
    -----
    arguments: list of strings
    
    Output
    -----
    compression_method : "lossless"
    compression_opts:  tuple
                       (backend:string, clevel:int) 
    """

    return parse_blosc_compression_options(arguments)


def parse_lossy_compression_options(arguments: List[str]):
    """
    Function to parse compression options for the lossy case
    Input
    -----
    arguments: list of strings
    
    Output
    -----
    compression_method : string="lossy"
    compression_opts:  tuple
                       (backend:string, method:string, parameter:int or float)
    """

    # If the list of arguments provided has only one element, use a default
    # Probably a risky thing to do, maybe raising a warning can be worth.
    if len(arguments) == 1:
        print("WARNING: You are using lossy compression without specifying more parameters."
              "This results in using zfp with rate mode at 4 bits per value.")
        return "lossy", ("zfp", "rate", 4)
    else:
        # Right now we only have the zfp case:
        if arguments[1] == "zfp":
            return parse_zfp_compression_options(arguments)
        elif arguments[1] == "sz":
            return parse_sz_compression_options(arguments)

        # If we want to add a different compressor should be as simple as:
        # elif arguments[1] == "compressor_name":
        #    return parse_compressor_name_compression_options(arguments)
        #
        # to keep things organized, this function will be declared in a different file, one for each compressor.
        #
        else:
            raise NotImplementedError("Compression: Unknown lossy compressor: %s" % arguments[1])


def parse_compression_configuration_file(filename: str) -> dict:
    """
    Function to parse a compression configuration file
    Input
    -----
    filename: str
        path fo the file that contains compression specifications.

    Output
    -----
    specifications : dict
    """
    # Look for file format:
    if filename.count(".json"):
        file_format = "json"
    elif filename.count(".yaml"):
        file_format = "yaml"
    else:
        raise AssertionError("Unknown configuration file format, expecting json or yaml")

    if file_format == "json":
        import json
        load_function = json.loads
    elif file_format == "yaml":
        import yaml

        def load_function(stream):
            return yaml.load(stream, yaml.SafeLoader)
    else:
        raise AssertionError("Unknown configuration file format! Supported formats are json and yaml.")

    with open(filename) as f:
        specifications = load_function(f.read())

    # Replace "None" strings with None values
    for key, value in specifications.items():
        if value == "None":
            specifications[key] = None
    return specifications
