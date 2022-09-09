"""
Routines for Evaluation of model forecasts
"""
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
