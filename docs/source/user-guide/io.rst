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


Compression
-----------

If **enstools-compression** is installed, it is possible to write compressed files, using lossless or lossy compressors.

For **lossless** compression it relies on `BLOSC <www.blosc.org>`_ and for **lossy** compression it relies on `ZFP <computing.llnl.gov/projects/zfp>`_ and `SZ <szcompressor.org/>`_.

Some examples
.. code::

    with read("input.nc") as dataset:
        ...
        # Lossless compression with the default parameters
        write(dataset, "output.nc", compression="lossless")
        ...
        # Lossy compression using SZ with the absolute threshold of 0.001
        write(dataset, "output.nc", compression="lossy,sz,abs,0.001")

        # Lossy compression using ZFP with the absolute threshold of 0.001
        write(dataset, "output.nc", compression="lossy,zfp,accuracy,0.001")

For more details about the compression please visit the `corresponding documentation <enstools-compression.readthedocs.io>`_.


See API
-------

:ref:`io-api`.