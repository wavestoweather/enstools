"""
Reading and Writing of meteorological data
"""
from .file_type import get_file_type
from .reader import read
from .writer import write
from .dataset import drop_unused
from .compressor import launch_compress_from_command_line, compress
from .analyzer import launch_analysis_from_command_line, analyze
from .cli import main as cli
