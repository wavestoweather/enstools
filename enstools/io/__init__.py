"""
Reading and Writing of meteorological data
"""


def __clean_HDF5_PLUGIN_PATH():
    """
    if the libraries from hdf5plugin are in HDF5_PLUGIN_PATH, then remove them
    """
    import os
    import logging
    if "HDF5_PLUGIN_PATH" in os.environ:
        paths = os.environ["HDF5_PLUGIN_PATH"].split(":")
        keep = []
        for one_path in paths:
            if len(one_path) == 0:
                continue
            if 'h5z-sz' not in one_path:
                logging.info(f"removed {one_path} from HDF5_PLUGIN_PATH")
                continue
            keep.append(one_path)
        if len(keep) > 0:
            os.environ["HDF5_PLUGIN_PATH"] = ":".join(keep)
        else:
            del os.environ["HDF5_PLUGIN_PATH"]


# TODO: figure out why this is needed and remove it!
__clean_HDF5_PLUGIN_PATH()

from .file_type import get_file_type
from .reader import read
from .writer import write
from .dataset import drop_unused
from enstools.io.compression.compressor import compress
from enstools.io.compression.analyzer import analyze
from enstools.io.compression.evaluator import evaluate
from enstools.io.compression.significant_bits import analyze_file_significant_bits
from .cli import main as cli

