Input/Output
==============

enstools.io provides two main functions:
    - **read** (:meth:`enstools.io.read`): Read netCDF or grib files and return an xarray.Dataset.
    - **write** (:meth:`enstools.io.write`): Write an xarray.Dataset into a netCDF file

A basic example:

.. code::

    with read("input.nc") as dataset:
        ...
        write(dataset, "output.nc")




If **enstools-compression** is installed, it is possible to write compressed files, using lossless or lossy compressors.
Check :ref:`Compression` for more details.


See API
-------

:ref:`io-api`.