#!/usr/bin/env python3

from enstools.opendata import retrieve, DWDContent

content = DWDContent()
print("Available Models:", content.get_models())
print("Available init times:", content.get_avail_init_times(model="icon"))
print("Available variables for icon and init_time 00:", content.get_avail_vars(model="icon", init_time=0))


retrieve(variable=["t_2m"],
         model="icon-eu",
         level_type="single",
         init_time=0,
         forecast_hour=[0, 1, 2, 3, 4],
         dest="downloads",
         merge_files=True)



# Faster access for multiple downloads:
print("Faster access...")
content.retrieve(variable=["t"],
                 level_type="pressure",
                 init_time=0,
                 levels=[1000, 950, 900],
                 forecast_hour=[0, 123],
                 dest="downloads",
                 merge_files=False)

content.retrieve(variable=["t"],
                 model="ICON-EU",
                 level_type="pressure",
                 init_time=0,
                 levels=[1000, 950, 900],
                 forecast_hour=[0],
                 dest="downloads",
                 merge_files=False)

content.retrieve(variable=["t"],
                 model="cosmo-d2",
                 grid_type="regular-lat-lon",
                 level_type="pressure",
                 init_time=0,
                 levels=[1000, 950, 975],
                 forecast_hour=[0],
                 dest="downloads",
                 merge_files=True)


# This is an invalid request:
retrieve(variable=["td"],  # Variable td is not available in icon-eps
         model="icon-eps",
         level_type="pressure",
         init_time=0,
         forecast_hour=[0, 1, 2, 3, 4],
         dest="downloads",
         merge_files=False)


# This is an invalid request:
retrieve(variable=["t"],
         model="ICON-EU",
         level_type="pressure",
         init_time=0,
         levels=[1000, 951, 900],  # level 951 is not available supposed to fail
         forecast_hour=[0],
         dest="downloads",
         merge_files=False)
