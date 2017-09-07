import os
import logging
import urllib
import bz2


def download(url, destination, uncompress=True):
    """
    Download a file from the web

    Parameters
    ----------
    url : string
            the web address

    destination : string
            path to store the file. The file will only be downloaded once.

    uncompress : bool
            if True and if the name ends with '.bz2', it will
            automatically be uncompressed.
    """
    if destination.endswith(".bz2") and uncompress:
        destination_intern = destination[:-4]
    else:
        destination_intern = destination

    if os.path.exists(destination_intern):
        logging.warning("file not downloaded because it is already present: %s" % url)
        return

    # download
    fn, hd = urllib.urlretrieve(url, destination)

    # uncompress
    if destination.endswith(".bz2") and uncompress:
        # uncompress
        bfile = bz2.BZ2File(destination)
        dfile = open(destination_intern, "wb")
        dfile.write(bfile.read())
        dfile.close()
        bfile.close()
        # delete compressed
        os.remove(destination)
