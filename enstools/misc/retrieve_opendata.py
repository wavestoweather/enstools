import os
from datetime import datetime
from enstools.misc import download
from enstools.io import read
import pandas
import logging


class DWDContent:

    def __init__(self, refresh_content=False):

        def create_dataframe(logdata):
            logging.info("Creating content database with {}".format(logdata))
            content = pandas.read_csv(logdata, delimiter="|", header=None, names=["file", "size", "time"])
            content["time"] = content["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

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
                "filename"].apply(lambda x: x.split("_")[6])
            content.loc[content["level_type"] == "model", ["level"]] = content[content.level_type == "model"][
                "filename"].apply(lambda x: x.split("_")[6])
            content["level"] = content["level"].astype(int)
            content.to_pickle("opendata_dwd_content.pkl")

            return content

        if os.path.exists("opendata_dwd_content.pkl") and not refresh_content:
            logging.info("Reading content database from opendata_dwd_content.pkl")
            content_old = pandas.read_pickle("opendata_dwd_content.pkl")
            self.content = content_old

        else:
            if refresh_content:
                logging.info("Refreshing content database")
            else:
                logging.info("Initializing content database")
            if os.path.exists("content.log"):
                os.remove("content.log")

            download("https://opendata.dwd.de/weather/nwp/content.log.bz2",
                     destination="content.log.bz2", uncompress=True)
            self.content = create_dataframe("content.log")

    def refresh_content(self):
        self.__init__(refresh_content=True)

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
                                    & (content["init_time"] == init_time)
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
                           & (content["forecast_hour"] == forecast_hour)
                           & (content["level"] == level)]["filename"].values[0]

        return filename

    def check_parameters(self, model=None, init_time=None, variable=None, level_type=None,
                         forecast_hour=None, levels=None):
        params_available = True
        if model not in self.get_models():
            params_available = False

        if init_time not in self.get_avail_init_times(model=model):
            params_available = False

        for var in variable:
            avail_vars = self.get_avail_vars(model=model, init_time=init_time)
            if var not in avail_vars:
                params_available = False

            avail_level_types = self.get_avail_level_types(model=model, init_time=init_time, variable=var)
            if level_type not in avail_level_types:
                params_available = False

            for hour in forecast_hour:
                if hour not in self.get_avail_forecast_hours(model=model, init_time=init_time,
                                                             variable=var, level_type=level_type):
                    params_available = False

                for lev in levels:
                    avail_levels = self.get_avail_levels(model=model, init_time=init_time,
                                                         variable=var, level_type=level_type)
                    if lev not in avail_levels:
                        params_available = False
        if not params_available:
            logging.warning("Parameters not available or database outdated" +
                            ", trying with refreshing the content database")
            self.refresh_content()

            avail_models = self.get_models()
            if model not in avail_models:
                raise ValueError("The model {} is not available. Possible Values: {}".format(model, avail_models))

            avail_init_times = self.get_avail_init_times(model=model)
            if init_time not in avail_init_times:
                raise ValueError("The initial time {} is not available. Possible Values: {}"
                                 .format(init_time, avail_init_times))

            for var in variable:
                avail_vars = self.get_avail_vars(model=model, init_time=init_time)
                if var not in avail_vars:
                    raise ValueError(
                        "The variable {} is not available for the {} model and the init_time {}. Available variables:{}"
                        .format(var, model, init_time, avail_vars))

                avail_level_types = self.get_avail_level_types(model=model, init_time=init_time, variable=var)
                if level_type not in avail_level_types:
                    raise ValueError("The level type {} is not available for the variable {}. Available types: {}"
                                     .format(level_type, var, avail_level_types))
                avail_forecast_hours = self.get_avail_forecast_hours(model=model, init_time=init_time,
                                                                     variable=var, level_type=level_type)
                for hour in forecast_hour:

                    if hour not in avail_forecast_hours:
                        raise ValueError("The forecast hour {} is not available for the variable {}. Possible values:{}"
                                         .format(hour, var, avail_forecast_hours))
                    for lev in levels:
                        avail_levels = self.get_avail_levels(model=model, init_time=init_time,
                                                             variable=var, level_type=level_type)
                        if lev not in avail_levels:
                            raise ValueError("The level {} is not available for the variable {} and the level type {}."
                                             .format(lev, var, level_type)
                                             + " Possible Values: {}".format(avail_levels))

    def retrieve_opendata(self, service="DWD", model="ICON", eps=None, variable=None, level_type=None, levels=0,
                          init_time=None, forecast_hour=None, merge_files=False, dest=None):
        """
        Downloads datasets from opendata server. Faster access to the database.
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

        init_time : int or str

        forecast_hour : list or str
                hours since the initialization of forecast. Multiple values are allowed.

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
        model = model.lower()
        if model.endswith("eps") and eps is False:
            raise ValueError("{} is a ensemble forecast, but eps was set to False!".format(model))
        elif eps is True and not model.endswith("-eps"):
            model = model + "-eps"

        download_files = []
        download_urls = []

        # Difference between DWDContent.retrieve_opendata() and retrieve_opendata():
        self.check_parameters(model=model, init_time=init_time, variable=variable, level_type=level_type,
                              forecast_hour=forecast_hour, levels=levels)

        for var in variable:
            for hour in forecast_hour:
                for lev in levels:
                    download_urls.append(self.get_url(model=model, init_time=init_time, variable=var,
                                                         level_type=level_type, forecast_hour=hour, level=lev))
                    download_files.append(self.get_filename(model=model, init_time=init_time, variable=var,
                                                               level_type=level_type, forecast_hour=hour, level=lev))

        download_files = [dest + "/" + file[:-4] for file in download_files]

        for i in range(len(download_urls)):
            download(download_urls[i], download_files[i] + ".bz2", uncompress=True)

        if merge_files:
            merge_dataset_name = dest + "/" + service + "_" + model + "_" \
                                 + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%S%fs") + ".nc"

            concat(download_files, merge_dataset_name)
            for file in download_files:
                os.remove(file)

        return download_files


def concat(files, out_filename):
    """
    Concatenates multiple files to one.
    Parameters
    ----------
    files: list
        The list of the files to concat
    out_filename: str
        The name (with destination) of the merged file

    Returns
    -------

    """
    out = open(out_filename, "wb")
    for filename in files:
        file = open(filename, "rb")
        out.write(file.read())
        file.close()
    out.close()

def retrieve_opendata(service="DWD", model="ICON", eps=None, variable=None, level_type=None, levels=0,
                      init_time=None, forecast_hour=None, merge_files=False, dest=None):
    """
    Downloads datasets from opendata server.
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

    init_time : int or str

    forecast_hour : list or str
            hours since the initialization of forecast. Multiple values are allowed.

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
    model = model.lower()
    if model.endswith("eps") and eps is False:
        raise ValueError("{} is a ensemble forecast, but eps was set to False!".format(model))
    elif eps is True and not model.endswith("-eps"):
        model = model + "-eps"

    download_files = []
    download_urls = []

    content = DWDContent()
    content.check_parameters(model=model, init_time=init_time, variable=variable, level_type=level_type,
                             forecast_hour=forecast_hour, levels=levels)

    for var in variable:
        for hour in forecast_hour:
            for lev in levels:
                download_urls.append(content.get_url(model=model, init_time=init_time, variable=var,
                                                     level_type=level_type, forecast_hour=hour, level=lev))
                download_files.append(content.get_filename(model=model, init_time=init_time, variable=var,
                                                           level_type=level_type, forecast_hour=hour, level=lev))

    download_files = [dest + "/" + file[:-4] for file in download_files]

    for i in range(len(download_urls)):
            download(download_urls[i], download_files[i] + ".bz2", uncompress=True)

    if merge_files:
        merge_dataset_name = dest + "/" + service + "_" + model + "_" \
                             + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%S%fs")+ ".nc"

        concat(download_files, merge_dataset_name)
        for file in download_files:
            os.remove(file)

    return download_files
