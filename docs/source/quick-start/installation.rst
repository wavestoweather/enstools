Installation
============


Python Package Index
--------------------
**Ensemble Tools** has `its own package in the Python Package Index <https://pypi.org/project/enstools/>`_.

Therefore, the easiest way of installing enstools is using **pip**
    >>> pip install enstools


Installing the latest version from github
-----------------------------------------

The packages available through the Python Package Index are not automatically updated.
In case of needing a newer version of the code, **pip** can install sources directly from a git repository.

To install the latest version from github:

    >>> pip install git+https://github.com/wavestoweather/enstools

To install a specific commit (for example 592a07919a5b9ef896d5810b3be6b9a45ac6bfb1):

    >>> pip install git+https://github.com/wavestoweather/enstools@592a07919a5b9ef896d5810b3be6b9a45ac6bfb1


For development
---------------
The sources of **Ensemble Tools** can be found in `its github repository <https://github.com/wavestoweather/enstools/>`_

A convenient way to install enstools for development is to make a local copy of the repository and install it as an
editable package:
    >>> git clone https://github.com/wavestoweather/enstools
    >>> pip install -e enstools/


System Dependencies
-------------------

To use enstools we need:

    - Python3
    - GEOS: required by **cartopy**

        In **Ubuntu** it can be installed by doing:
            >>> apt install libgeos-dev

    - eccodes (Optional): If present, enstools will be able to read **.grib** files.

        In **Ubuntu** it can be installed by doing:
            >>> apt install libeccodes-dev


Workaround for GEOS
^^^^^^^^^^^^^^^^^^^

.. _GEOS: https://libgeos.org

In case the user has no root permissions and don't manage to install `GEOS`_ it is possible to install enstools
**without cartopy**. To do that:

    1. Install enstools without its dependencies:
        >>> pip install enstools --no-deps

    2. Get list of missing dependencies and ignore **cartopy**:
        >>> pip check | cut -d "," -f 1 | cut -d " " -f 4 | grep -v "cartopy" > pending_requirements.txt

    3. Install the pending requirements:
        >>> pip install -r pending_requirements.txt

This will allow the installation of **enstools** but
it won't be possible to use the features that require cartopy (enstools.plot).
