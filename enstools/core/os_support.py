"""
some service functions for interaction with OS environments
"""
import six
import os
import socket
from contextlib import closing
if six.PY2:
    from commands import getstatusoutput
else:
    from subprocess import getstatusoutput


def getenv(name):
    """
    get the value of an environment variable.

    Parameters
    ----------
    name : str
            name of the environment variable

    Raises
    ------
    KeyError :
            raises a KeyError if the Variable was not found in the environment

    Returns
    -------
    the value
    """
    value = os.getenv(name)
    if value is None:
        raise KeyError("Environment Variable not found: %s" % name)
    return value


def get_first_free_port(host, start_port):
    """
    find a free port on the given host on which the scheduler can be started

    Parameters
    ----------
    host
    start_port

    Returns
    -------
    int :
            the first free port above the start_port
    """
    while True:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind((host, start_port))
            except:
                start_port +=1
                continue
            return s.getsockname()[1]


def which(cmd):
    """
    use the unix command which to find an executable

    Parameters
    ----------
    cmd : str
            name of the executable to search

    Returns
    -------
    str:
            absolute path of the executable
    """
    sts, out = getstatusoutput("which %s" % cmd)
    if sts != 0:
        raise IOError("executable not found: %s" % cmd)
    return out
