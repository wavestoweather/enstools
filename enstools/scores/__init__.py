"""
Routines for Evaluation of model forecasts
"""
# Imports for typing
from typing import Callable
from xarray import DataArray

# Import actual scores
from .DisplacementAmplitudeScore import das
from .continuous_ranked_probability_score import continuous_ranked_probability_score
from .kolmogorov_smirnov import kolmogorov_smirnov, kolmogorov_smirnov_index
from .kolmogorov_smirnov_multicell import kolmogorov_smirnov_multicell
from .normalized_root_mean_square_error import (mean_square_error,
                                                root_mean_square_error,
                                                normalized_root_mean_square_error,
                                                normalized_root_mean_square_error_index,
                                                )
from .pearson_correlation import pearson_correlation, pearson_correlation_index
from .structural_similarity_index import structural_similarity_index, structural_similarity_log_index
from .peak_signal_to_noise_ratio import peak_signal_to_noise_ratio
from .positivity import positivity

# Add some aliases
ssim_I = structural_similarity_log_index
correlation_I = pearson_correlation_index
nrmse_I = normalized_root_mean_square_error_index
ks_I = kolmogorov_smirnov_index
psnr = peak_signal_to_noise_ratio


def add_score_from_file(file_path: str):
    """
    Read a python file and add the function with the same name as the file to the available scores in enstools.scores
    """

    import importlib
    import os
    module_name = os.path.basename(file_path).strip(".py")

    import importlib.util
    import sys

    # Get spec from the file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    # Get the module
    mod = importlib.util.module_from_spec(spec)

    # Load the module
    spec.loader.exec_module(mod)

    # Get the function
    function = mod.__getattribute__(module_name)

    register_score(function, module_name)


def register_score(function: Callable[[DataArray, DataArray], DataArray], name: str):
    """
    Register a given function as a score.
    """
    import sys
    # Add the function as a new score in module enstools.scores
    setattr(sys.modules[__name__], name, function)
