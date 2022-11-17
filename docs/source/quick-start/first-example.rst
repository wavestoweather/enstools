First example
=============

A small example

.. code::

    # Import
    import enstools.io

    # Define file name
    input_file = "input.nc"

    # Open file as an xarray dataset
    with enstools.io.read(input_file) as dataset:
        # Loop over the different variables
        for variable in dataset.data_vars:
           print(dataset[variable])
