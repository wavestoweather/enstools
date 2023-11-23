.. _enstools-opendata:

Open Data
==============



Ensemble Tools provide few functions to download meteorological data from the opendata website provided by `DWD <https://www.dwd.de>`_.

Command Line Interface
----------------------

.. autoprogram:: enstools.opendata.cli:get_parser()
    :prog: enstools-compression

Python API
----------

This small example downloads total precipitation data from ICON-EU for 0,6,12,18 and 24 forecast hours.

.. code::

    from enstools.opendata import retrieve_nwp

    # download the ICON-EU 24h forecast from today 00 UTC.
    grib_file = retrieve_nwp(variable=["tot_prec"],
                             model="icon-eu",
                             grid_type="regular-lat-lon",
                             level_type="single",
                             init_time=0,
                             forecast_hour=[0,6,12,18,24],
                             dest="data",
                             merge_files=True,
                             validate_urls=True)



for more details: :ref:`opendata-api`.