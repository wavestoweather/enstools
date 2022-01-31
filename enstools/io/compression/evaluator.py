"""
#
# Functions to evaluate differences in netcdf/grib files that should represent the same data.
#

"""
from enstools.io.compression.metrics import DataArrayMetrics, DatasetMetrics


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


v_char = u'\u2713'


def print_green(text: str):
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")


def print_red(text: str):
    print(f"{bcolors.FAIL}{text}{bcolors.ENDC}")


def evaluate(reference_path: str, target_path: str, plot: bool = False, create_gradients=False):
    """
    The purpose of this routine is to obtain some metrics and plots on how similar are two datasets.
    """

    file_comparison = DatasetMetrics(reference_path, target_path)

    if create_gradients:
        # Compute gradients and add it as new variables
        file_comparison.create_gradients()
        # Compute second order gradients and add it as new variables
        file_comparison.create_second_order_gradients()

    # Get list of variables
    variables = file_comparison.variables

    # As a tentative idea, we can rise some warnings in case some metrics are below certain thresholds:
    # These thresholds could be:
    #   ssim_I < 3
    #   correlation_I < 4
    #   nrmse_I < 2

    def checks(metrics: DataArrayMetrics):
        thresholds = {
            "ssim_I": 3,
            "correlation_I": 4,
            "nrmse_I": 2,
            # "max_rel_diff": 10000000,
            "ks_I": 2,
        }
        for key, value in thresholds.items():
            if metrics[key] < value:
                yield f"{bcolors.BOLD}{key}{bcolors.ENDC} index is low: {metrics[key]:.1f}."

    warnings = {}
    for variable in variables:
        warnings[variable] = list(checks(file_comparison[variable]))

    for variable in variables:
        if warnings[variable]:
            print_red(f"{variable} X")
            for warning in warnings[variable]:
                print(f"\t{warning}")
        else:
            print_green(f"{variable} {v_char}")

    print("\nSUMMARY:")
    num_variables_with_warnings = sum([1 if len(warnings[v]) > 0 else 0 for v in variables])
    if not num_variables_with_warnings:
        print_green(f"Any variable has warnings!")
    else:
        print(f"{num_variables_with_warnings}/{len(variables)}  variables have warnings.\n\n")

    # Produce visual reports
    if plot:
        file_comparison.make_plots()
