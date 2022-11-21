from pathlib import Path
from typing import Union, List
import glob


def clean_paths(paths: Union[Path, List[Path], List[str], str],
                check_files_exist: bool = True
                ,) -> List[Path]:
    """
    Take the paths in all the usual forms and convert them to a list of Path objects.
    Parameters
    ----------
    paths

    Returns
    -------

    """

    # Check arguments

    if not isinstance(paths, (list, str, tuple, Path)):
        raise NotImplementedError("Unsupported type of argument: %s" % type(paths))    

    if not isinstance(paths, list):
        paths = [paths]

    # Expand paths using glob
    expanded_paths = [glob.glob(str(fp)) for fp in paths]

    # Merge elements in a flat list
    expanded_paths = [item for sublist in expanded_paths for item in sublist]

    # Check that there are actually files.
    if not expanded_paths:
        raise FileNotFoundError(f"Files {str(paths)} don't exist.")

    # Convert the elements to Path and expand
    path_objects = [Path(fp).resolve() for fp in expanded_paths]
    if check_files_exist:
        for fp in path_objects:
            assert fp.exists()
    return path_objects
