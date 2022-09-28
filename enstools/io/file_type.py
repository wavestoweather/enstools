from enstools.core import getstatusoutput
import logging
import re
import os


def get_file_type(filename: str, only_extension=False):
    """
    use the first bytes to guess the type of input files

    Parameters
    ----------
    filename : string
            name of the file to check

    only_extension : bool
            if True, only the file name is checked. This is useful for not yet created files.

    Returns
    -------
    string
            NC, HDF, GRIB
    """
    # guess the type based on extension
    file_type_based_on_extension = None
    file_type = None
    basename, extension = os.path.splitext(filename)
    if extension in [".nc", ".nc3", ".nc4"]:
        file_type_based_on_extension = "NC"
    elif extension in [".grb", ".grib", ".grb2", ".grib2"]:
        file_type_based_on_extension = "GRIB"

    if not only_extension:
        # is the file present at all?
        if not os.path.exists(filename):
            raise IOError("file '%s' not found!" % filename)

        # read the first bytes and decide based on the content
        with open(filename, "rb") as f:
            first_bytes = f.read(12)

            # netcdf3 file
            if first_bytes.startswith(b"CDF"):
                file_type = "NC"
            # netcdf4
            elif b"HDF" in first_bytes[0:4]:
                file_type = "HDF"
            # grib
            elif b"GRIB" in first_bytes:
                file_type = "GRIB"

    # return the best knowledge of the file type
    if file_type is not None:
        return file_type
    else:
        return file_type_based_on_extension

