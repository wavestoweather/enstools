from enstools.core import check_arguments
import numpy as np

from .nearest_neighbour_interpolator import nearest_neighbour


@check_arguments(shape={"arr": (0, 0)})
def downsize(arr, fac):
    """
    Reduce resolution of an array by neighbourhood averaging - 2D averaging of fac x fac element

    Parameters
    ----------
    arr : xarray.DataArray or np.ndarray
            array to downsize by neighbourhood averaging

    fac : int
            factor of downsizing, 2D averaging of fac x fac element

    Returns
    -------
    xarray.DataArray or np.ndarray
    """
    row, col = np.shape(arr)
    r_small = int(row // fac)
    c_small = int(col // fac)
    return np.column_stack(np.column_stack(
        arr.reshape((r_small, row // r_small, c_small, col // c_small)))).mean(1).reshape((r_small, c_small))


