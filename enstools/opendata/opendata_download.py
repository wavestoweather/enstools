import logging
import os
import pandas
from datetime import datetime
from enstools.misc import download, bytes2human, concat
from urllib.error import HTTPError
from urllib.request import urlopen


class DWDContent:

    def __init__(self, refresh_content=False):

        def create_dataframe(logdata):
            """
            Creates a pandas.DataFrame from the given logdata from https://opendata.dwd.de/weather/nwp
            and sets the following attributes:
                file: str
                    Name of the url of the file.
                size: str
                    The size of the file.
                time: np.datetime64
                    The creation time on the server.
                model: str
                    The forecast model, e.g. "icon".
                file_type: str
                    The format of the file ("grib" or "json").
                init_time: int
                    The time of the initialisation of the forecast for the given file.
                variable: str
                    The short name of the variable of the file.
                filename: str
                    The name of the file.
                level_type: str
                    The type of the level of the file.
                forecast_hour: int
                    Hours since the initalization of the forecast.

            Parameters
            ----------
            logdata: str
                The name of the content logfile.

            Returns
            -------
            content: pandas.DataFrame
                The DataFrame object with the given attributes.

            """
            logging.info("Creating content database with {}".format(logdata))
            content = pandas.read_csv(logdata, delimiter="|", header=None, names=["file", "size", "time"])
            content["time"] = content["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            content["size"] = content["size"].astype(int)

            content = content[~content.file.str.contains("snow4")]
            content = content[~content.file.str.contains("content.log")]

            # fix for json data, no json support now:
            content = content[~content.file.str.contains("json")]

            content["model"] = content["file"].apply(lambda x: x.split("/")[1])
            content["file_type"] = content["file"].apply(lambda x: x.split("/")[2])
            content["init_time"] = content["file"].apply(lambda x: x.split("/")[3])
            content["init_time"] = content["init_time"].astype(int)
            content["variable"] = content["file"].apply(lambda x: x.split("/")[4])
            content["filename"] = content["file"].apply(lambda x: x.split("/")[5])
            content["file"] = content["file"].apply(lambda x: "https://opendata.dwd.de/weather/nwp" + x[1:])

            content["level_type"] = content["filename"].apply(lambda x: x.split("_")[3])
            content["grid_type"] = content["filename"].apply(lambda x: x.split("_")[2])
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
        """
        Initializes the DWDContent object again.
        Downloads the actual content.log from the server and creates a new DataFrame
        """
        self.__init__(refresh_content=True)

    def get_models(self):
        """
        Gives the available models of the data server.
        Returns
        -------
        avail_models: list
            List of Strings with the available models

        """
        content = self.content
        avail_models = content["model"].drop_duplicates().values.tolist()
        avail_models.sort()
        return avail_models

    def get_grid_types(self, model=None):
        """
        Gives the available grid_types for a specific model
        Parameters
        ----------
        model: str
            The model of the forecast simulation.

        Returns
        -------

        """
        content = self.content
        grid_types = content[content["model"] == model]["grid_type"].drop_duplicates().values.tolist()
        return grid_types

    def check_grid_type(self, model=None, grid_type=None):
        """
        Checks if the given grid_type is available and if grid_type=None check if the available type is unambigious.

        Parameters
        ----------
        model: str
            The model of the forecast.
        grid_type: str
            The geo grid type.

        Returns
        -------
        grid_type: str
            if available and unambigious the right grid type will be returned

        """
        avail_grid_types = self.get_grid_types(model=model)
        if grid_type is None:
            if len(avail_grid_types) is 1:
                grid_type = avail_grid_types[0]
            else:
                raise ValueError("You have to choose one of the grid_types: {}".format(avail_grid_types))
        else:
            if grid_type not in avail_grid_types:
                raise ValueError("Grid type {} not available for model {}. available grid_type: {}"
                                 .format(grid_type, model, avail_grid_types))
        return grid_type

    def check_level_type(self, model=None, grid_type=None, init_time=None, variable=None, level_type=None):
        """
        Checks if the given level_type is available for the variable in the given model. If no level_type is given,
        and only one is available, then the one will be returned.

        Parameters
        ----------
        model: str
            The model of the forecast.
        grid_type: str
            The geo grid type.
        init_time: int
            The initialization time.
        variable: list or tuple
            The variable list
        level_type

        Returns
        -------

        """
        avail_level_types = []
        for var in variable:
            alevlist = self.get_avail_level_types(model=model, grid_type=grid_type, init_time=init_time, variable=var)
            # No duplicate values:
            for alev in alevlist:
                if alev not in avail_level_types:
                    avail_level_types.append(alev)
        if level_type is None:
            if len(avail_level_types) is 1:
                level_type = avail_level_types[0]

            else:
                raise ValueError("You have to choose one of the level_types: {}".format(avail_level_types))
        else:
            if level_type not in avail_level_types and len(avail_level_types) is not 0:
                raise ValueError("Level type {} not available for model {}, variable {}. Available level_type: {}"
                                 .format(level_type, model, variable, avail_level_types))
        return level_type

    def get_avail_init_times(self, model=None, grid_type=None):
        """
        Gives the available initialization times for a given forecast model.

        Parameters
        ----------
        model: str
            The model for which the available initialization times want to be known.
        grid_type: str
            The type of the geo grid.

        Returns
        -------
        avail_init_times: list
            A list of integers of the available initialization times

        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content
        avail_init_times = content[(content["model"] == model)
                                   & (content["grid_type"] == grid_type)]["init_time"].drop_duplicates().values.tolist()
        avail_init_times.sort()
        return avail_init_times

    def get_avail_vars(self, model=None, grid_type=None, init_time=None):
        """
        Gives the available variables for a given forecast model and initialization time.

        Parameters
        ----------
        model: str
            The model for which the available variables want to be known.
        grid_type: str
            The type of the geo grid.
        init_time: int
            The initialization time for which the available variables want to be known.

        Returns
        -------
        avail_vars: list
            The sorted list of strings of the available variables.

        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content
        avail_vars = content[(content["model"] == model)
                             & (content["grid_type"] == grid_type)
                             & (content["init_time"] == init_time)]["variable"].drop_duplicates().values.tolist()
        avail_vars.sort()
        return avail_vars

    def get_avail_level_types(self, model=None, grid_type=None, init_time=None, variable=None):
        """
        Gives the available level types for a given forecast model, initialization time and variable.
        Parameters
        ----------
        model: str
            The model for which the available level types want to be known.
        grid_type: str
            The type of the geo grid.
        init_time: int
            The initialization time for which the available level types want to be known.
        variable: str
            The variable for which the available level types want to be known

        Returns
        -------
        avail_level_types: list
            The sorted list of strings of the available level types.

        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content
        avail_level_types = content[(content["model"] == model)
                                    & (content["grid_type"] == grid_type)
                                    & (content["init_time"] == init_time)
                                    & (content["variable"] == variable)]["level_type"].drop_duplicates().values.tolist()
        avail_level_types.sort()
        return avail_level_types

    def get_avail_forecast_hours(self, model=None, grid_type=None, init_time=None, variable=None, level_type=None):
        """
        Gives the available  forecast hours since initialization for a given forecast model, initialization time,
        variable and level type.

        Parameters
        ----------
        model: str
            The model for which the available forecast hours want to be known.
        grid_type: str
            The type of the geo grid.
        init_time: int
            The initialization time for which the available forecast hours want to be known.
        variable: str
            The variable for which the available forecast hours want to be known.
        level_type: str
            The type of the level for which the available forecast hours want to be known.

        Returns
        -------
        avail_forecast_hours: list
            The available hours of the forecast data since the initilization of the forecast.

        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content
        avail_forecast_times = content[(content["model"] == model)
                                       & (content["grid_type"] == grid_type)
                                       & (content["init_time"] == init_time)
                                       & (content["variable"] == variable)
                                       & (content["level_type"] == level_type)]["forecast_hour"]\
            .drop_duplicates().values.tolist()
        avail_forecast_times.sort()
        return avail_forecast_times

    def get_avail_levels(self, model=None, grid_type=None, init_time=None, variable=None, level_type=None):
        """
        Gives the available levels since initialization for a given forecast model, initialization time,
        variable and level type. If the level of the variable is not pressure or model 0 will be returned.

        Parameters
        ----------
        model: str
            The model for which the available levels want to be known.
        grid_type: str
            The type of the geo grid.
        init_time: int
            The initialization time for which the available levels want to be known.
        variable: str
            The variable for which the available forecast hours want to be known.
        level_type: str
            The type of the level for which the available levels want to be known.

        Returns
        -------
        avail_levels: list
            A list of integers of the available levels.

        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content
        avail_levels = content[(content["model"] == model)
                               & (content["grid_type"] == grid_type)
                               & (content["init_time"] == init_time)
                               & (content["variable"] == variable)
                               & (content["level_type"] == level_type)]["level"].drop_duplicates().values.tolist()
        avail_levels.sort()
        return avail_levels

    def get_url(self, model=None, grid_type=None, init_time=None, variable=None,
                level_type=None, forecast_hour=None, level=None):
        """
        Gives the url of the file on the https://opendata.dwd.de/weather/nwp server.

        Parameters
        ----------
        model: str
            The model of the file for which the url wants to be known.
        grid_type: str
            The type of the geo grid.
        init_time: int
            The initialization time of the forecast of the file for which the url wants to be known.
        variable: str
            The variable of the file for which the url wants to be known.
        level_type: str
            The type of level of the file for which the url wants to be known.
        forecast_hour: int
            The hours since the initialization of the forecast of the file for which the url wants to be known.
        level: int
            The level of the file for which the url wants to be known.

        Returns
        -------
        url: str
            The url adress of the file.
        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content

        url = content[(content["model"] == model)
                      & (content["init_time"] == init_time)
                      & (content["variable"] == variable)
                      & (content["level_type"] == level_type)
                      & (content["forecast_hour"] == forecast_hour)
                      & (content["level"] == level)
                      & (content["grid_type"] == grid_type)]["file"].values
        if len(url) != 1:

            raise KeyError("Url not found(model:{}, grid_type{}, init_time:{},variable:{},level_type:{}"
                           ",forecast_hour:{},level:{})"
                           .format(model, grid_type, init_time, variable, level_type, forecast_hour, level))

        return url.item()

    def get_filename(self, model=None, grid_type=None, init_time=None, variable=None,
                     level_type=None, forecast_hour=None, level=None):
        """
         Gives the filename of the file on the https://opendata.dwd.de/weather/nwp server.

        Parameters
        ----------
        model: str
            The model of the file for which the filename wants to be known.
        grid_type: str
            The type of the geo grid.
        init_time: int
            The initialization time of the forecast of the file for which the filename wants to be known.
        variable: str
            The variable of the file for which the filename wants to be known.
        level_type: str
            The type of level of the file for which the filename wants to be known.
        forecast_hour: int
            The hours since the initialization of the forecast of the file for which the filename wants to be known.
        level: int
            The level of the file for which the filename wants to be known.

        Returns
        -------
        url: str
            The filename of the file.
        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content
        filename = content[(content["model"] == model)
                           & (content["init_time"] == init_time)
                           & (content["variable"] == variable)
                           & (content["level_type"] == level_type)
                           & (content["forecast_hour"] == forecast_hour)
                           & (content["level"] == level)
                           & (content["grid_type"] == grid_type)]["filename"].values.item()

        return filename

    def check_url_available(self, model=None, grid_type=None, init_time=None, variable=None,
                            level_type=None, forecast_hour=None, level=None):
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        try:
            http_code = urlopen(self.get_url(model=model, grid_type=grid_type, init_time=init_time, variable=variable,
                                             level_type=level_type, forecast_hour=forecast_hour, level=level)).getcode()
            if http_code == 200:
                url_available = True
        except HTTPError:
            url_available = False
        return url_available

    def get_file_size(self, model=None, grid_type=None, init_time=None, variable=None,
                      level_type=None, forecast_hour=None, level=None):
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content
        size = content[(content["model"] == model)
                       & (content["init_time"] == init_time)
                       & (content["variable"] == variable)
                       & (content["level_type"] == level_type)
                       & (content["forecast_hour"] == forecast_hour)
                       & (content["level"] == level)
                       & (content["grid_type"] == grid_type)]["size"].values.item()
        return size

    def get_size_of_download(self, model=None, grid_type=None, init_time=None,
                             variable=None, level_type=None, forecast_hour=None, levels=None):
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        total_size = 0
        for var in variable:
            for hour in forecast_hour:
                for lev in levels:
                    total_size += self.get_file_size(model=model, grid_type=grid_type, init_time=init_time,
                                                     variable=var, level_type=level_type, forecast_hour=hour, level=lev)
        return total_size

    def check_parameters(self, model=None, grid_type=None, init_time=None, variable=None,
                         level_type=None, forecast_hour=None, levels=None):
        """
        Checks if there are all files available for the given parameters.
        If not, the DWDContent object will be refreshed.
        If one file is not available for the given parameters, a detailed error will be thrown.

        Parameters
        ----------
        model:str
            The model of the file.
        grid_type: str
            The type of the geo grid.
        init_time: int
            The initialization time of the file.
        variable: list or str
            The variable of the file.
        level_type: str
            The type of level of the file.
        forecast_hour: list or int
            The hours of the forecast since the initialization of the simulation.
        levels: list or int
            The levels.

        Returns
        -------

        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)

        params_available = True
        if model not in self.get_models():
            params_available = False

        if init_time not in self.get_avail_init_times(model=model, grid_type=grid_type):
            params_available = False

        for var in variable:
            avail_vars = self.get_avail_vars(model=model, grid_type=grid_type, init_time=init_time)
            if var not in avail_vars:
                params_available = False
                break

            avail_level_types = self.get_avail_level_types(model=model, grid_type=grid_type,
                                                           init_time=init_time, variable=var)
            if level_type not in avail_level_types:
                params_available = False
                break

            for hour in forecast_hour:
                if hour not in self.get_avail_forecast_hours(model=model, grid_type=grid_type, init_time=init_time,
                                                             variable=var, level_type=level_type):
                    params_available = False
                    break

                for lev in levels:
                    avail_levels = self.get_avail_levels(model=model, grid_type=grid_type, init_time=init_time,
                                                         variable=var, level_type=level_type)

                    if lev not in avail_levels:
                        params_available = False
                        break

        if not params_available:
            logging.warning("Parameters not available")
        else:
            for hour in forecast_hour:
                for var in variable:
                    for lev in levels:
                        if not self.check_url_available(model=model, grid_type=grid_type, init_time=init_time,
                                                        variable=var, level_type=level_type, forecast_hour=hour,
                                                        level=lev):
                            params_available = False
                            break
            if not params_available:
                logging.warning("Database outdated.")

        if not params_available:
            self.refresh_content()

            avail_models = self.get_models()
            if model not in avail_models:
                raise ValueError("The model {} is not available. Possible Values: {}".format(model, avail_models))

            avail_init_times = self.get_avail_init_times(model=model, grid_type=grid_type)
            if init_time not in avail_init_times:
                raise ValueError("The initial time {} is not available. Possible Values: {}"
                                 .format(init_time, avail_init_times))

            for var in variable:
                avail_vars = self.get_avail_vars(model=model, grid_type=grid_type, init_time=init_time)
                if var not in avail_vars:
                    raise ValueError(
                        "The variable {} is not available for the {} model and the init_time {}. Available variables:{}"
                        .format(var, model, init_time, avail_vars))

                avail_level_types = self.get_avail_level_types(model=model, grid_type=grid_type,
                                                               init_time=init_time, variable=var)
                if level_type not in avail_level_types:
                    raise ValueError("The level type {} is not available for the variable {}. Available types: {}"
                                     .format(level_type, var, avail_level_types))
                avail_forecast_hours = self.get_avail_forecast_hours(model=model, grid_type=grid_type,
                                                                     init_time=init_time, variable=var,
                                                                     level_type=level_type)
                for hour in forecast_hour:

                    if hour not in avail_forecast_hours:
                        raise ValueError("The forecast hour {} is not available for the variable {}. Possible values:{}"
                                         .format(hour, var, avail_forecast_hours))
                    for lev in levels:
                        avail_levels = self.get_avail_levels(model=model, grid_type=grid_type, init_time=init_time,
                                                             variable=var, level_type=level_type)
                        if lev not in avail_levels:
                            raise ValueError("The level {} is not available for the variable {} and the level type {}."
                                             .format(lev, var, level_type)
                                             + " Possible Values: {}".format(avail_levels))

    def get_merge_dataset_name(self, model=None, variable=None,
                               level_type=None, init_time=None):
        merge_name = "merge_" + model + "_" + level_type + "_" + "init" + str(init_time) + "_"

        for var in variable:
            merge_name = merge_name + var + "+"
        merge_name = merge_name[:-1]

        return merge_name + "_" + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%S%fs") + ".grib2"

    def retrieve_opendata(self, service="DWD", model="ICON", eps=None, grid_type=None, variable=None, level_type=None,
                          levels=0, init_time=None, forecast_hour=None, merge_files=False, dest=None):
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

        grid_type: str
            The type of the geo grid.

        variable : list or str
                list of variables to download. Multiple values are allowed.

        level_type : str
                one of "model", "pressure", or "single"

        levels : list or int
                levels to download. Unit depends on `level_type`.

        init_time : int or str

        forecast_hour : list or int
                hours since the initialization of forecast. Multiple values are allowed.

        merge_files : bool
                if true, GRIB files are concatenated to create one file.

        dest : str
                Destination folder for downloaded data. If the files are already available,
                they are not downloaded again.

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
        if service.lower() != "dwd":
            raise KeyError("Only DWD server is available")

        if not os.path.exists(dest):
            os.mkdir(dest)
        model = model.lower()
        if grid_type is not None:
            grid_type = grid_type.lower()
        if level_type is not None:
            level_type = level_type.lower()
        variable = [var.lower() for var in variable]

        if model.endswith("eps") and eps is False:
            raise ValueError("{} is a ensemble forecast, but eps was set to False!".format(model))
        elif eps is True and not model.endswith("-eps"):
            model = model + "-eps"

        download_files = []
        download_urls = []

        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        level_type = self.check_level_type(model=model, grid_type=grid_type, init_time=init_time,
                                           variable=variable, level_type=level_type)

        self.check_parameters(model=model, grid_type=grid_type, init_time=init_time, variable=variable,
                              level_type=level_type, forecast_hour=forecast_hour, levels=levels)

        for var in variable:
            for hour in forecast_hour:
                for lev in levels:
                    download_urls.append(self.get_url(model=model, grid_type=grid_type, init_time=init_time,
                                                      variable=var, level_type=level_type, forecast_hour=hour,
                                                      level=lev))
                    download_files.append(self.get_filename(model=model, grid_type=grid_type, init_time=init_time,
                                                            variable=var, level_type=level_type, forecast_hour=hour,
                                                            level=lev))

        download_files = [dest + "/" + file[:-4] for file in download_files]

        total_size_human = bytes2human(self.get_size_of_download(model=model, grid_type=grid_type, init_time=init_time,
                                                                 variable=variable, level_type=level_type,
                                                                 forecast_hour=forecast_hour, levels=levels))
        logging.info("Downloading {} files with the total size of {}".format(len(download_files), total_size_human))
        for i in range(len(download_urls)):
            download(download_urls[i], download_files[i] + ".bz2", uncompress=True)

        if merge_files:
            merge_dataset_name = dest + "/" + self.get_merge_dataset_name(model=model, variable=variable,
                                                                          level_type=level_type, init_time=init_time)

            concat(download_files, merge_dataset_name)
            for file in download_files:
                os.remove(file)
            return merge_dataset_name

        else:
            return download_files


def retrieve_opendata(service="DWD", model="ICON", eps=None, grid_type=None, variable=None, level_type=None,
                      levels=0, init_time=None, forecast_hour=None, merge_files=False, dest=None):
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

    grid_type: str
        The type of the geo grid.

    variable : list or str
            list of variables to download. Multiple values are allowed.

    level_type : str
            one of "model", "pressure", or "single"

    levels : list or int
            levels to download. Unit depends on `level_type`.

    init_time : int or str

    forecast_hour : list or int
            hours since the initialization of forecast. Multiple values are allowed.

    merge_files : bool
            if true, GRIB files are concatenated to create one file.

    dest : str
            Destination folder for downloaded data. If the files are already available,
            they are not downloaded again.

    Returns
    -------
    list :
            names of downloaded files.
    """
    content = DWDContent()
    download_files = content.retrieve_opendata(service=service, model=model, eps=eps, grid_type=grid_type,
                                               variable=variable, level_type=level_type, levels=levels,
                                               init_time=init_time, forecast_hour=forecast_hour,
                                               merge_files=merge_files, dest=dest)
    return download_files
