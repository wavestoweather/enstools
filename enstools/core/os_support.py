"""
some service functions for interaction with OS environments
"""
import six
import os
import re
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


def get_ip_address(interface=None):
    """
    Get an IP-Address to be used for dask communication. If an interface is given, the IP of this interface is used.
    Otherwise the fastest interface will be used.

    Parameters
    ----------
    interface : str
            Name of the interface to be used.

    Returns
    -------
    str:
            IP-Address to be used for communication.
    """
    # use socket module to get the IP associated with the hostname
    ip_hostname = socket.gethostbyname(socket.gethostname())

    # get a list of all interfaces if none was specified. The list is stored in a dictionary which will transport
    # all information about one interface
    interfaces = {}
    if interface is None:
        sts, out = getstatusoutput("ip link show up")
        if sts != 0:
            raise OSError("unable to get list of interfaces: %s" % out)
        # extract a list of all interfaces from the output
        interface_lines = re.split("^\d+: ", out, flags=re.MULTILINE)
        for line in interface_lines[1:]:
            iface = line.split(":", 1)[0]
            interfaces[iface] = {}
    else:
        interfaces[interface] = {}
    logging.debug("get_ip_address: list of interfaces: %s" % ", ".join(interfaces.keys()))

    # get the IPs for all interfaces
    for one_interface in interfaces.keys():
        sts, out = getstatusoutput("ip addr show %s" % one_interface)
        if sts != 0:
            raise OSError("unable to the ip address for interface %s" % one_interface)
        addr_match = re.search("inet (\d+.\d+.\d+.\d+)", out)
        if addr_match is None:
            logging.debug("get_ip_address: no IP address found for interface %s" % one_interface)
            interfaces[one_interface]["ip"] = None
        else:
            interfaces[one_interface]["ip"] = addr_match.group(1)
        logging.debug("get_ip_address: IP for interface %s: %s" % (one_interface, interfaces[one_interface]["ip"]))

    # find the fastest interface
    fastest_speed = -9999
    fastest_interfaces = None
    logging.debug("get_ip_address: speed of interfaces:")
    for one_interface in interfaces.keys():
        try:
            with open("/sys/class/net/%s/speed" % one_interface, "r") as f:
                interfaces[one_interface]["speed"] = int(f.read())
        except IOError:
            interfaces[one_interface]["speed"] = 0
        if interfaces[one_interface]["speed"] == fastest_speed:
            fastest_interfaces.append(one_interface)
        if interfaces[one_interface]["speed"] > fastest_speed:
            fastest_speed = interfaces[one_interface]["speed"]
            fastest_interfaces = [one_interface]
        logging.debug("get_ip_address: interfaces %s: %d" % (one_interface, interfaces[one_interface]["speed"]))

    # if there are more interfaces with the same speed, check if one has the IP associated with the hostname
    fastest_interface = fastest_interfaces[0]
    if len(fastest_interfaces) > 1:
        logging.debug("get_ip_address: found %d interfaces with same speed:" % len(fastest_interfaces))
        for iface in fastest_interfaces:
            logging.debug("get_ip_address: %s, IP: %s" % (iface, interfaces[iface]["ip"]))
            if interfaces[iface]["ip"] is not None and ip_hostname == interfaces[iface]["ip"]:
                logging.debug("get_ip_address: %s is associated with the hostname, selecting this interface!" % (iface))
                fastest_interface = iface
                break

    # log selection
    logging.debug("get_ip_address: selected interface:  %s" % fastest_interface)
    logging.debug("get_ip_address: selected IP-address: %s" % interfaces[fastest_interface]["ip"])
    return interfaces[fastest_interface]["ip"]


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
        self.oep = None
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
            line = line.rstrip()
            # INFO output from dask processes should only be shown in debug mode.
            if b"- INFO -" in line:
                logging.debug(line)
            else:
                print(line)
        logging.debug("%s exited with exit code %d" % (self.args[0], self.p.returncode))
        if self.p.returncode != 0:
            logging.debug(self.p.stdout.read())

        # if there is an command specified to be executed at the end of the thread, then do it now
        if self.on_exit_args is not None:
            self.oep = Popen(self.on_exit_args)
            self.oep.wait()

    def poll(self):
        """
        the the process state. If an on exit process is defined, poll that as well.
        """
        p_poll = self.p.poll()
        if self.oep is not None:
            oep_poll = self.oep.poll()
        else:
            oep_poll = 0
        if p_poll is None:
            return None
        else:
            if oep_poll is None:
                return None
            else:
                return p_poll + oep_poll

    def terminate(self):
        return self.p.terminate()

    @property
    def returncode(self):
        return self.p.returncode

