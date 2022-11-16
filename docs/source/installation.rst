Installation
============

Python Package Index
____________________
The easiest way of installing enstools is using **pip**
    >>> pip install enstools

Dependencies
____________
Besides the dependencies that are handled by **pip**, the `GEOS package<https://libgeos.org/>`_ is required by **cartopy**.
In **Ubuntu** it can be installed by doing:
    >>> apt install libgeos-dev

Workaround for GEOS
...................
In case the user has no root permissions and can't manage to install the `GEOS package<https://libgeos.org/>`_ it is possible to install enstools
without cartopy. To do that:

    1. Install enstools without its dependencies:
        >>> pip install enstools --no-deps

    2. Get list of missing dependencies and ignore **cartopy**:
        >>> pip check | cut -d "," -f 1 | cut -d " " -f 4 | grep -v "cartopy" > pending_requirements.txt

    3. Install the pending requirements:
        >>> pip install -r pending_requirements.txt

This will allow the installation of **enstools** but
 it won't be possible to use the features that require cartopy.

For development
---------------
A convenient way to install enstools for development is to make a local copy of the repository and install it as an
editable package:
    >>> git clone https://github.com/wavestoweather/enstools
    >>> pip install -e enstools/

