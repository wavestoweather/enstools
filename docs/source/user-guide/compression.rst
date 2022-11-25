.. _Compression:

Compression
====================

If the package **enstools-compression** is installed, the I/O :ref:`enstools-io` functions provided by enstools can use **lossless** and **lossy** compression.

Check more details about the package in:

    - `GitHub repository <https://github.com/wavestoweather/enstools-compression>`_ 
    - `Documentation <https://enstools-compression.readthedocs.io>`_ |compressionbadge|


.. |compressionbadge| image:: https://readthedocs.org/projects/enstools-compresion/badge/?version=latest
    :target: https://enstools-compression.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


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
