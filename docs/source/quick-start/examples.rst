Basic Example
=============

**Ensemble Tools** have many features. In this example we show how we can use it to download, plot and save weather data as a compressed netCDF.

.. code::

    
    from enstools.opendata import retrieve_nwp

    # download the ICON-EU 24h forecast from today 00 UTC.
    grib_file = retrieve_nwp(variable=["tot_prec"],
                             model="icon-eu",
                             grid_type="regular-lat-lon",
                             level_type="single",
                             init_time=0,
                             forecast_hour=[24],
                             dest=args.data,
                             merge_files=True)


    
    # read the grib file using enstools.io.read
    import enstools.io

    with enstools.io.read(grib_file) as dataset:

        # create a basic contour plot and show it
        import matplotlib.pyplot as plt
        import enstools.plot
        fig, ax = enstools.plot.contour(dataset["tp"][0, :, :], coastlines="50m")    
    
        plt.show()
    
    
        # Save the dataset using lossy compression
        enstools.io.write(dataset, "my_new_file.nc", compression="lossy,sz,pw_rel,1.e-5")

Explore the :ref:`UserGuide` to find more examples!