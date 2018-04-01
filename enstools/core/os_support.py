"""
some service functions for interaction with OS environments
"""
import six
import os
import socket
from contextlib import closing
from threading import Thread
from subprocess import Popen, PIPE, STDOUT
import logging
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


class ProcessObserver(Thread):
    """
    run a process and store its stdout
    """
    def __init__(self, args):
        """
        create an observer for a given command and start the process

        Parameters
        ----------
        cmd : str
                command to start and to follow
        """
        super(ProcessObserver, self).__init__()
        self.args = args
        self.on_exit_args = None
        self.p = None
        self.setDaemon(True)
        self.start()

    def run_on_exit(self, args):
        """
        an additional command to run after the main process ended (e.g. for cleanup)

        Parameters
        ----------
        args : list
                arguments of the program as expected by Popen
        """
        self.on_exit_args = args

    def run(self):
        """
        start the process and store it's output
        """
        # start the process
        self.p = Popen(self.args, stdin=PIPE, stdout=PIPE, stderr=STDOUT)

        # observe the output of the process as long as it is running
        while self.p.poll() is None:
            line = self.p.stdout.readline()
            if len(line) == 0:
                continue
            # INFO output from dask processes should only be shown in debug mode.
            if "- INFO -" in line:
                logging.debug(line)
            else:
                print(line)

        # if there is an command specified to be executed at the end of the thread, then do it now
        if self.on_exit_args is not None:
            oea = Popen(self.on_exit_args)
            oea.wait()

    def poll(self):
        return self.p.poll()

    def terminate(self):
        return self.p.terminate()

    @property
    def returncode(self):
        return self.p.returncode

