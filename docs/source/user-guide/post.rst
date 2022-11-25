Postprocessing
==============

.. warning::
    This documentation is incomplete.

.. code::

    # read the grib file
    dataset = enstools.io.read(grib_file)

    # variables names (depending on availability of DWD grib definitions)
    tp_name = "tp" if not "TOT_PREC" in grib else "TOT_PREC" 

    # calculate 3-hourly precipitation from accumulated values
    dataset3h = dataset.isel(time=slice(None, None, 12))
    tp3h = (dataset3h[tp_name][1:,...] - np.array(dataset3h[tp_name][0:-1,...])) / 3.0
    tp3h.attrs["units"] = "kg m-2 hour-1"
 
    # calculate 3-hourly mean values of cape, ignore last 45min
    cape3h = dataset["CAPE_ML"].resample(time="3H", label='right').reduce(np.mean)[0:-1,...]
    cape3h.attrs["units"] = "J/kg"
 
    # calculate convective adjustment timescale with default parameters
    tau = convective_adjustment_time_scale(pr=tp3h, cape=cape3h)


See API
-------

:ref:`post-api`.
