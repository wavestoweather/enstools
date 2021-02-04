"""
Reading and Writing of meteorological data
"""
from .file_type import get_file_type
from .reader import read
from .writer import write
from .dataset import drop_unused
from .compressor import compress
from .analyzer import find_compression_parameters
