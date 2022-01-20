#!/usr/bin/env python3

from enstools.opendata import retrieve_nwp, getDWDContent

# search for data using the content object
content = getDWDContent()
print("Available Models:", content.get_models())
print("Available init times:", content.get_avail_init_times(model="icon"))
print("Available variables for icon and init_time 00:", content.get_avail_vars(model="icon", init_time=0))


# examples for data download
retrieve_nwp(variable=["t_2m"],
             model="icon-eu",
             level_type="single",
             init_time=0,
             forecast_hour=[0, 1, 2, 3, 4],
             dest="data",
             merge_files=True)

retrieve_nwp(variable=["t"],
             level_type="pressure",
             init_time=0,
             levels=[1000, 950, 900],
             forecast_hour=[0, 123],
             dest="data",
             merge_files=False)

retrieve_nwp(variable=["t"],
             model="ICON-EU",
             level_type="pressure",
             init_time=0,
             levels=[1000, 950, 900],
             forecast_hour=[0],
             dest="data",
             merge_files=False)

try:
    retrieve_nwp(variable=["t"],
                 model="cosmo-d2",
                 grid_type="regular-lat-lon",
                 level_type="pressure",
                 init_time=0,
                 levels=[1000, 950, 975],
                 forecast_hour=[0],
                 dest="data",
                 merge_files=True)
except ValueError:
    print("The request of variable t for the model cosmo-d2 does no longer work")

try:
    # This is an invalid request:
    retrieve_nwp(variable=["td"],  # Variable td is not available in icon-eps
                 model="icon-eps",
                 level_type="pressure",
                 init_time=0,
                 forecast_hour=[0, 1, 2, 3, 4],
                 dest="data",
                 merge_files=False)
except ValueError:
    print("As expected, the request of the td variable for icon-eps failed")


# This is an invalid request:
try:
    retrieve_nwp(variable=["t"],
                 model="ICON-EU",
                 level_type="pressure",
                 init_time=0,
                 levels=[1000, 951, 900],  # level 951 is not available supposed to fail
                 forecast_hour=[0],
                 dest="data",
                 merge_files=False)
except ValueError:
    print("As expected, the request of the t variable at level 951 for icon-eu failed")
