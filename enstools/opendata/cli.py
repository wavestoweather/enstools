"""
Command line interface for enstools.opendata. Currently only data from DWD is available.
"""
import os
import argparse
import logging
from datetime import datetime
from enstools.opendata import retrieve, getDWDContent

# default format used on the command line.
__DATE_FORMAT="%Y-%m-%dT%H:%M"


def error_exit(text):
    """
    quit with error code 1

    Parameters
    ----------
    text: str
            error message to show before quiting.
    """
    logging.error(text)
    exit(1)


def retrieve_data(args):
    """
    use enstools.opendata.retrieve to download files

    Parameters
    ----------
    args:
            command-line arguments
    """
    if args.model is None:
        error_exit("--model is required for retrieve!")
    if args.grid_type is None:
        error_exit("--grid_type is required for retrieve!")
    if args.init_time is None:
        error_exit("--init-time is required for retrieve!")
    if args.variable is None:
        error_exit("--variable is required for retrieve!")
    if args.level_type is None:
        error_exit("--level-type is required for retrieve!")
    if args.levels is None:
        error_exit("--levels is required for retrieve!")
    if args.lead_time is None:
        error_exit("--lead-time is required for retrieve!")
    if args.dest is None:
        error_exit("--dest is required for retrieve!")
    if not os.path.exists(args.dest):
        error_exit(f"path not found: {args.dest}")
    if not os.path.isdir(args.dest):
        error_exit(f"path is not a directory: {args.dest}")
    try:
        files = retrieve(model=args.model,
                         grid_type=args.grid_type,
                         variable=args.variable,
                         level_type=args.level_type,
                         levels=args.levels,
                         init_time=args.init_time,
                         forecast_hour=args.lead_time,
                         dest=args.dest,
                         merge_files=args.merge)
        for one_file in files:
            logging.info(f"downloaded: {one_file}")
    except ValueError as ve:
        error_exit(str(ve))

def query(args):
    """
    find files on opendata platforms

    Parameters
    ----------
    args:
            command-line arguments
    """
    content = getDWDContent()

    # list all available models
    if args.get_models:
        logging.info(f"Available models: {', '.join(content.get_models())}")

    # the available init times for a specific model/grid
    if args.get_init_times:
        if args.model is None:
            error_exit("--model is required for this query!")
        try:
            init_times = content.get_avail_init_times(args.model, args.grid_type)
            init_times_str = []
            for one_time in init_times:
                if type(one_time) == int:
                    init_times_str.append(str(one_time))
                elif isinstance(one_time, datetime):
                    init_times_str.append(one_time.strftime(__DATE_FORMAT))
            logging.info(f"Available init times: {', '.join(init_times_str)}")
        except ValueError as ve:
            error_exit(str(ve))

    # the available variables for a model, grid and init time
    if args.get_vars:
        if args.model is None:
            error_exit("--model is required for this query!")
        if args.init_time is None:
            error_exit("--init-time is required for this query!")
        try:
            logging.info(f"Available variables: {', '.join(content.get_avail_vars(model=args.model, grid_type=args.grid_type, init_time=args.init_time))}")
        except ValueError as ve:
            error_exit(str(ve))

    if args.get_level_types:
        if args.model is None:
            error_exit("--model is required for this query!")
        if args.init_time is None:
            error_exit("--init-time is required for this query!")
        if args.variable is None:
            error_exit("--variable is required for this query!")
        try:
            logging.info(f"Available level-types: {', '.join(content.get_avail_level_types(model=args.model, grid_type=args.grid_type, init_time=args.init_time, variable=args.variable[0]))}")
        except ValueError as ve:
            error_exit(str(ve))

    if args.get_levels:
        if args.model is None:
            error_exit("--model is required for this query!")
        if args.init_time is None:
            error_exit("--init-time is required for this query!")
        if args.variable is None:
            error_exit("--variable is required for this query!")
        if args.level_type is None:
            error_exit("--level-type is required for this query!")
        try:
            levels = content.get_avail_levels(model=args.model, grid_type=args.grid_type, init_time=args.init_time, variable=args.variable[0], level_type=args.level_type)
            levels = list(map(lambda x: str(x), levels))
            logging.info(f"Available levels: {', '.join(levels)}")
        except ValueError as ve:
            error_exit(str(ve))

    if args.get_lead_times:
        if args.model is None:
            error_exit("--model is required for this query!")
        if args.init_time is None:
            error_exit("--init-time is required for this query!")
        if args.variable is None:
            error_exit("--variable is required for this query!")
        if args.level_type is None:
            error_exit("--level-type is required for this query!")
        try:
            hours = content.get_avail_forecast_hours(model=args.model, grid_type=args.grid_type, init_time=args.init_time, variable=args.variable[0], level_type=args.level_type)
            hours = list(map(lambda x: f"{x}h", hours))
            logging.info(f"Available forecast lead-times: {', '.join(hours)}")
        except ValueError as ve:
            error_exit(str(ve))


def __add_common_args(parser: argparse.ArgumentParser):
    """
    add common arguments to sub-parsers
    """
    parser.add_argument("--model", help="name of the model to use. Use query --get-models to get a list of valid names.")
    parser.add_argument("--grid-type", help="type of the grid to use.")
    parser.add_argument("--level-type", help="type of the vertical level to use.")
    parser.add_argument("--init-time", help=f"initialization time to use. "
                                            "Integers are interpreted as hours since model start, dates formatted as "
                                            f"{__DATE_FORMAT.replace('%Y', 'YYYY').replace('%m', 'MM').replace('%d', 'DD').replace('%H', 'HH').replace('%M', 'MM')} are interpreted as absolute start dates.")
    parser.add_argument("--variable", nargs="+", help="name of the variable to use. Use query --get-vars to get a list of valid names.")
    parser.add_argument("--levels", nargs="+", type=int, help="levels to use.")
    parser.add_argument("--lead-time", nargs="+", type=int, help="lead times to use in hours.")


def __parse_init_time(args):
    """
    is the command line arguments contain an init time, then it is here converted to datetime
    """
    if args.init_time is None:
        return
    try:
        if args.init_time.isdigit():
            args.init_time=int(args.init_time)
        else:
            args.init_time=datetime.strptime(args.init_time, __DATE_FORMAT)
    except Exception as ex:
        error_exit(str(ex))


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(help="Searching and downloading data is available as sub-commands.")

    # sub-command for downloading data
    parser_retrieve = subparsers.add_parser("retrieve", help="download files from an opendata service provider")
    parser_retrieve.add_argument("--dest", help="destination folder for downloaded data.")
    parser_retrieve.add_argument("--merge", action='store_true', help="merge multiple downloaded files.")
    __add_common_args(parser_retrieve)
    parser_retrieve.set_defaults(func=retrieve_data)

    # sub-command for finding data
    parser_query = subparsers.add_parser("query", help="search for data on opendata service providers")
    parser_query.add_argument("--get-models", action='store_true', help="list available models.")
    parser_query.add_argument("--get-init-times", action='store_true', help="list available init times. Requires argument --model and accepts --grid-type.")
    parser_query.add_argument("--get-vars", action='store_true', help="list of available variables for --model, --grid-type, and --init-time.")
    parser_query.add_argument("--get-level-types", action='store_true', help="get the available types of vertical levels for --model, --grid-type, --init-time, and --variable.")
    parser_query.add_argument("--get-levels", action='store_true', help="get the available vertical levels for --model, --grid-type, --init-time, --variable, and --level-type.")
    parser_query.add_argument("--get-lead-times", action='store_true', help="get the available times since model start for --model, --grid-type, --init-time, and --level-type.")
    __add_common_args(parser_query)
    parser_query.set_defaults(func=query)

    args = parser.parse_args()
    __parse_init_time(args)
    args.func(args)
