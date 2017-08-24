from enstools.core import getstatusoutput
import logging
import re
import os

__file_command = None
__file_version = None


def get_file_type(filename):
    """
    use libmagic to guess the type of input files

    Parameters
    ----------
    filename : string
            name of the file to check

    Returns
    -------
    string
            NC, HDF, GRIB
    """
    # is the file present at all?
    if not os.path.exists(filename):
        raise IOError("file '%s' not found!" % filename)

    # find the file command
    global __file_command
    global __file_version
    if __file_command is None:
        sts, __file_command = getstatusoutput("which file")
        if sts != 0:
            __file_command = None
            __file_version = None
            logging.warning("the 'file' command is not available; file types are estimated based on extension only!")
        else:
            sts, __file_version = getstatusoutput("%s -v" % __file_command)
            try:
                __file_version = re.search("\d\.\d+", __file_version).group(0)
            except AttributeError:
                __file_command = None
                __file_version = None
                logging.warning("unable to read the version of the file command; file types are estimated based on extension only!")

    # guess the type based on extension
    file_type_based_on_extension = None
    file_type = None
    basename, extension = os.path.splitext(filename)
    if extension in [".nc", ".nc3", ".nc4"]:
        file_type_based_on_extension = "NC"
    elif extension in [".grb", ".grib", ".grb2", ".grib2"]:
        file_type_based_on_extension = "GRIB"

    # use the file command to find the type
    if __file_command is not None:
        cmd = "%s -b %s" % (__file_command, filename)
        sts, ft = getstatusoutput(cmd)
        if sts != 0:
            logging.warning("the command '%s' failed; file types are estimated based on extension only!" % cmd)
        else:
            if "Hierarchical Data Format" in ft:
                if file_type_based_on_extension is not None and "NC" in file_type_based_on_extension:
                    file_type = "NC"
                else:
                    file_type = "HDF"
            if "Gridded binary (GRIB)" in ft:
                file_type = "GRIB"

    # return the best knowledge of the file type
    if file_type is not None:
        return file_type
    else:
        return file_type_based_on_extension

