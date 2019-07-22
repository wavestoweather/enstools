import os
from urllib.request import urlopen
from datetime import datetime, timedelta
from enstools.misc import download

def retrieve_opendata(service="DWD", model="ICON", eps=False, variable=None, level_type=None, levels=None,
                      init_date=None, forecast_hour=None, merge_files=False, dest=None):
    """
    Parameters
    ----------
    service : str
            name of weather service. Default="DWD".

    model : str
            name of the model. Default="ICON".

    eps : bool
            if True, download ensemble forecast, otherwise download deterministic forecast.

    variable : list or str
            list of variables to download. Multiple values are allowed.

    level_type : str
            one of "model", "pressure", or "single"

    levels : list or range
            levels to download. Unit depends on `level_type`.

    init_date : list or str
            WIP: for icon init_date can only be a list of "00", "06", "12", "18".

    forecast_hour : list or string
            hours since the initialization of forecast. Multiple values are allowed.
            WIP: For icon only from "000" to "180" with step 3 allowed

    merge_files : bool
            if true, GRIB files are concatenated to create one file.

    dest : str
            Destination folder for downloaded data. If the files are already available, they are not downloaded again.

    Returns
    -------
    list :
            names of downloaded files.
    """
    # Want to download one or more variables?
    if not isinstance(variable, (list, tuple)):
        variable = [variable]
    # Want to download one or more forecast hours?
    if not isinstance(forecast_hour, (list, tuple)):
        forecast_hour = [forecast_hour]
    if not isinstance(levels, (list, tuple)):
        levels = [levels]

    downloaded_files = []

    if service == "DWD":
        root_url = "http://opendata.dwd.de/weather/nwp/"
        if model == "ICON":
            fc_times = ["00", "06", "12", "18"]

            vars = ["alb_rad", "alhfl_s", "asob_s", "asob_t", "aswdifd_s", "aswdifu_s", "aswdir_s", "cape_con",
                    "cape_ml", "clat", "clc", "clch", "clcl", "clcm", "clct", "clct_mod", "cldepth", "clon", "elat",
                    "elon", "fi", "fr_ice", "fr_lake", "fr_land", "hbas_con", "hhl", "h_snow", "hsurf", "htop_con",
                    "htop_dc", "hzerocl", "p", "plcov, pmsl", "ps", "qv", "qv_s", "rain_con", "rain_gsp",
                    "relhum", "relhum_2m", "rho_snow", "snow_con", "snow_gsp", "soiltyp", "t", "t_2m", "td_2m",
                    "t_g", "tke", "tmax_2m", "tmin_2m", "tot_prec", "t_snow", "t_so", "u", "u_10m", "v", "v_10m",
                    "vmax_10m", "w", "w_snow", "w_so", "ww", "z0"]

            model_vars = ["clc", "p", "qv", "t", "tke", "u", "v", "w"]
            pressure_vars = ["fi", "relhum", "t", "u", "v"]

            # No support:
            time_invariant_vars = ["clat", "clon", "elat", "elon", "fr_lake", "fr_land", "hhl", "hsurf", "plcov",
                                   "soiltyp"]
            soil_vars = ["t_so", "w_so"]

            if eps is False:
                if init_date in fc_times:
                    if int(init_date) > datetime.now().hour:
                        yesterday = datetime.now().date() - timedelta(days=1)
                        daystr = yesterday.strftime("%Y%m%d")
                    else:
                        daystr = datetime.now().date().strftime("%Y%m%d")
                else:
                    raise KeyError("Choose the initial date of the forecast between {} or a list of them"
                                   .format(fc_times))

                for hour in forecast_hour:

                    for var in variable:

                        if var not in vars:
                            raise KeyError("The variable {} is not available.".format(var))
                        if var in time_invariant_vars:
                            raise KeyError("The variable {} is not supported.".format(var))
                        if var in soil_vars:
                            raise KeyError("The variable {} is not supported.".format(var))

                        if level_type == "single":
                            if (var in pressure_vars) or (var in model_vars):
                                raise KeyError("The variable {} is not a single level variable.".format(var))

                            files = ["icon_global_icosahedral_single-level_" + daystr + init_date
                                     + "_" + hour + "_" + var.upper() + ".grib2"]

                        elif level_type == "pressure":
                            if var not in pressure_vars:
                                raise KeyError("The variable {} is not a pressure level variable.".format(var))
                            # Source for pressure levels
                            # https://www.dwd.de/SharedDocs/downloads/DE/modelldokumentationen/nwv/icon/
                            # icon_dbbeschr_aktuell.pdf?view=nasPublication&nn=13934
                            # page 31
                            pressure_levels = [1000, 950, 925, 900, 850, 800, 700, 600, 500,
                                               400, 300 , 250, 200, 150, 100 , 70, 50, 30]
                            for level in levels:
                                if level not in pressure_levels:
                                    raise KeyError("The pressure level {} is not available. Possible Values: {}"
                                                   .format(level, pressure_levels))

                            files = ["icon_global_icosahedral_pressure-level_" + daystr + init_date + "_"
                                    + hour + "_" + str(level) + "_" + var.upper() + ".grib2" for level in levels]

                        elif level_type == "model":
                            if var not in model_vars:
                                raise KeyError("The variable {} is not a model level variable.".format(var))

                            model_levels = list(range(1,91))

                            for level in levels:
                                if level not in model_levels:
                                    raise KeyError("The model level {} is not available. Possible Values: {}"
                                                   .format(level, model_levels))

                            files = ["icon_global_icosahedral_model-level_" + daystr + init_date + "_"
                                     + hour + "_" + str(level) + "_" + var.upper() + ".grib2" for level in levels]

                        else:
                            raise KeyError("Choose between 'model', 'pressure' or 'single'.")

                        url_path = root_url + "icon/grib/" + init_date + "/" + var + "/"

                        file_urls = [url_path + file + ".bz2" for file in files]

                        # Check if all files exist, before starting to download:
                        for url in file_urls:
                            if urlopen(url).getcode() != 200:
                                raise Exception("internal Error")

                        for i in range(len(file_urls)):
                            download(file_urls[i], dest+"/"+files[i]+".bz2", uncompress=True)
                            downloaded_files.append(dest+"/"+files[i])

    if merge_files:
        merge_dataset =  read([file for file in downloaded_files])
        merge_dataset_name = dest + "/" + service + "_" + model + "_" \
                             + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%S%fs") + ".nc"
        merge_dataset.to_netcdf(merge_dataset_name)
        for file in downloaded_files:
            os.remove(file)

    return downloaded_files

retrieve_opendata(variable=["t"],
                  level_type="pressure",
                  init_date="00",
                  levels=[1000,950,900],
                  forecast_hour = ["000", "123"],
                  dest = "dl",
                  merge_files = True)

