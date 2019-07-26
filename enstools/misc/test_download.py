import os
from urllib.request import urlopen
from datetime import datetime, timedelta
from enstools.misc import download
from enstools.io import read
import pandas
import filecmp

class DWD_Content():
    def __init__(self, destination):

        def create_dataframe(logdata):
            content = pandas.read_csv(logdata, delimiter="|", header=None, names=["file", "size", "time"])
            content = content[~content.file.str.contains("snow4")]
            content = content[~content.file.str.contains("content.log")]

            content["model"] = content["file"].apply(lambda x: x.split("/")[1])
            content["file_type"] = content["file"].apply(lambda x: x.split("/")[2])
            content["init_time"] = content["file"].apply(lambda x: x.split("/")[3])
            content["init_time"] = content["init_time"].astype(int)
            content["variable"] = content["file"].apply(lambda x: x.split("/")[4])
            content["filename"] = content["file"].apply(lambda x: x.split("/")[5])
            content["file"] = content["file"].apply(lambda x: "https://opendata.dwd.de/weather/nwp" + x[1:])

            content["level_type"] = content["filename"].apply(lambda x: x.split("_")[3])
            content["level_type"] = content["level_type"].apply(lambda x: x[:-6])

            content["forecast_hour"] = content["filename"].apply(lambda x: x.split("_")[5])
            content.loc[content["level_type"] == "time-inv", ["forecast_hour"]] = "0"
            content["forecast_hour"] = content["forecast_hour"].astype(int)

            content["level"] = "0"
            content.loc[content["level_type"] == "single", ["level"]] = "0"
            content.loc[content["level_type"] == "pressure", ["level"]] = content[content.level_type == "pressure"][
                "filename"].apply(
                lambda x: x.split("_")[6])
            content.loc[content["level_type"] == "model", ["level"]] = content[content.level_type == "model"][
                "filename"].apply(
                lambda x: x.split("_")[6])
            content["level"] = content["level"].astype(int)
            content.to_pickle("content.pkl")

            return content

        if os.path.exists("content.log"):
            os.rename("content.log", "content_old.log")
            download("https://opendata.dwd.de/weather/nwp/content.log.bz2", destination="content.log.bz2",
                     uncompress=True)
            if filecmp.cmp("content.log", "content_old.log"):
                self.content = pandas.read_picke("content.pkl")
            else:
                self.content = create_dataframe("content.log")

        else:
            download("https://opendata.dwd.de/weather/nwp/content.log.bz2",destination="content.log.bz2", uncompress=True)
            self.content = create_dataframe("content.log")





    def get_models(self):
        content = self.content
        avail_models = content["model"].drop_duplicates().values.tolist()
        avail_models.sort()
        return avail_models

    def get_avail_init_times(self, model=None):
        content = self.content
        avail_init_times = content[content["model"] == model]["init_time"].drop_duplicates().values.tolist()
        avail_init_times.sort()
        return avail_init_times

    def get_avail_vars(self, model=None, init_time=None):
        content = self.content
        avail_vars = content[(content["model"] == model)
                              & (content["init_time"] == init_time)]["variable"].drop_duplicates().values.tolist()
        avail_vars.sort()
        return avail_vars

    def get_avail_level_types(self, model=None, init_time=None, variable=None):
        content = self.content
        avail_level_types = content[(content["model"] == model)
                                    & (content["init_time"] ==init_time)
                                    & (content["variable"] == variable)]["level_type"].drop_duplicates().values.tolist()
        avail_level_types.sort()
        return avail_level_types

    def get_avail_forecast_hours(self, model=None, init_time=None, variable=None, level_type=None):
        content = self.content
        avail_forecast_times = content[(content["model"] == model)
                                       & (content["init_time"] == init_time)
                                       & (content["variable"] == variable)
                                       & (content["level_type"] == level_type)]["forecast_hour"]\
            .drop_duplicates().values.tolist()
        avail_forecast_times.sort()
        return avail_forecast_times

    def get_avail_levels(self, model=None, init_time=None, variable=None, level_type=None):
        content = self.content
        avail_levels = content[(content["model"] == model)
                               & (content["init_time"] == init_time)
                               & (content["variable"] == variable)
                               & (content["level_type"] == level_type)]["level"].drop_duplicates().values.tolist()
        avail_levels.sort()
        return avail_levels

    def get_url(self, model=None, init_time=None, variable=None, level_type=None, forecast_hour=None, level=None):
        content = self.content

        url = content[(content["model"] == model)
                      & (content["init_time"] == init_time)
                      & (content["variable"] == variable)
                      & (content["level_type"] == level_type)
                      & (content["forecast_hour"] == forecast_hour) & (content["level"] == level)]["file"].values

        return url[0]

    def get_filename(self, model=None, init_time=None, variable=None, level_type=None, forecast_hour=None, level=None):
        content = self.content
        filename = content[(content["model"] == model)
                           & (content["init_time"] == init_time)
                           & (content["variable"] == variable)
                           & (content["level_type"] == level_type)
                           & (content["forecast_hour"] == forecast_hour) & (content["level"] == level)]["filename"].values[0]

        return filename


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

    if not os.path.exists(dest):
        os.mkdir(dest)

    download_files = []
    download_urls = []
    model = model.lower()
    content = DWD_Content(dest)


    if model not in content.get_models():
        raise KeyError("The model {} is not available. Possible Values: {}".format(model, content.get_models()))

    if init_date not in content.get_avail_init_times(model=model):
        raise KeyError("The initial time {} is not available. Possible Values: {}"
                       .format(init_date, content.get_avail_init_times(model=model)))

    for var in variable:
        if var not in content.get_avail_vars(model=model, init_time=init_date):
            raise KeyError("The variable is not available for the {} model and the init_time {}. Available variables:{}"
                           .format(model, init_date, content.get_avail_vars(model=model, init_time=init_date)))
    for var in variable:
        avail_level_types = content.get_avail_level_types(model=model, init_time=init_date, variable=var)
        if level_type not in avail_level_types:
            raise KeyError("The level type {} is not available for the variable {}. Available types: {}"
                           .format(level_type, var, avail_level_types))

        for hour in forecast_hour:
            if hour not in content.get_avail_forecast_hours(model=model, init_time=init_date,
                                                            variable=var, level_type=level_type):
                raise KeyError("The forecast hour {} is not available for the variable {}. Possible values:{}"
                               .format(hour, var, content.get_avail_forecast_hours(model=model,
                                                                                   init_time=init_date,
                                                                                   variable=var,
                                                                                   level_type=level_type)))
            for lev in levels:
                avail_levels = content.get_avail_levels(model=model, init_time=init_date,
                                                       variable=var, level_type=level_type)
                if lev not in avail_levels:
                    raise KeyError("The level {} is not available for the variable {} and the level type {}."
                                   .format(lev, var, level_type)
                                   + " Possible Values: {}".format(avail_levels))
                print(model, init_date, var, level_type, forecast_hour, lev)
                print(content.get_url(model=model, init_time=init_date, variable=var,
                                      level_type=level_type, forecast_hour=hour, level=lev))
                download_urls.append(content.get_url(model=model, init_time=init_date, variable=var,
                                                     level_type=level_type, forecast_hour=hour, level=lev))
                download_files.append(content.get_filename(model=model, init_time=init_date, variable=var,
                                                     level_type=level_type, forecast_hour=hour, level=lev))

    download_files = [dest + "/" + file[:-4] for file in download_files]

    for i in range(len(download_urls)):
        download(download_urls[i], download_files[i], uncompress=True)

    if merge_files:
        merge_dataset = read([file for file in download_files])
        merge_dataset_name = dest + "/" + service + "_" + model + "_" \
                             + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%S%fs") + ".nc"
        merge_dataset.to_netcdf(merge_dataset_name)
        for file in download_files:
            os.remove(file)

    return download_files


retrieve_opendata(variable=["t"],
                  level_type="pressure",
                  init_date=0,
                  levels=[1000, 950, 900],
                  forecast_hour=[0, 123],
                  dest="dl",
                  merge_files=True)

retrieve_opendata(variable=["t"],
                  model = "ICON-EU",
                  level_type="pressure",
                  init_date=0,
                  levels=[1000, 950, 900],
                  forecast_hour=[0],
                  dest="dl",
                  merge_files=True)
