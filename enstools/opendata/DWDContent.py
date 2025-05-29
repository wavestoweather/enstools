import logging
import os
import time
import pandas
import itertools
import hashlib
import requests
from datetime import datetime
from enstools.misc import download, bytes2human, concat
from urllib.error import HTTPError
from urllib.request import urlopen
from enstools.core import get_cache_dir


class DWDContent:
    # create a new temporal directory to store content.log and opendata_dwd_content.pkl
    cache_path = get_cache_dir()

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
            content["time"] = pandas.to_datetime(content["time"], format="%Y-%m-%d %H:%M:%S")
            content["size"] = content["size"].astype(int)

            content = content[~content.file.str.contains("snow4")]
            content = content[~content.file.str.contains("content.log")]

            # fix for json data, no json support now:
            content = content[~content.file.str.contains("json")]

            # Helper functions that catch out of range array accessing.
            def get_string_segment(input, idx, delim):
                splitted = input.split(delim)
                if (len(splitted) < idx + 1):
                    return ""
                return splitted[idx]
            
            def to_int_or_zero(input):
                try:
                    return int(input)
                except:
                    return 0

            content["model"] = content["file"].apply(lambda x: get_string_segment(x, 1, '/'))
            content["file_type"] = content["file"].apply(lambda x: get_string_segment(x, 2, '/'))

            # Init time: Convert safely to int type.
            content["init_time"] = content["file"].apply(lambda x: get_string_segment(x, 3, '/'))
            content["init_time"] = content["init_time"].apply(to_int_or_zero)
            content["init_time"] = content["init_time"].astype(int)
            
            content["variable"] = content["file"].apply(lambda x: get_string_segment(x, 4, '/'))
            content["filename"] = content["file"].apply(lambda x: get_string_segment(x, 5, '/'))
            content["file"] = content["file"].apply(lambda x: "https://opendata.dwd.de/weather/nwp" + x[1:])

            content["abs_init_time"] = content["filename"].apply(lambda x: get_string_segment(x, 4, '_'))
            content["level_type"] = content["filename"].apply(lambda x: get_string_segment(x, 3, '_'))
            content["grid_type"] = content["filename"].apply(lambda x: get_string_segment(x, 2, '_'))
            content["level_type"] = content["level_type"].apply(lambda x: x[:-6])

            content["forecast_hour"] = content["filename"].apply(lambda x: get_string_segment(x, 5, '_'))
            content.loc[content["level_type"] == "time-inv", ["forecast_hour"]] = "0"
            content["forecast_hour"] = content["forecast_hour"].apply(to_int_or_zero)
            content["forecast_hour"] = content["forecast_hour"].astype(int)

            content["level"] = "0"
            content.loc[content["level_type"] == "single", ["level"]] = "0"
            content.loc[content["level_type"] == "pressure", ["level"]] = \
                    content[content.level_type == "pressure"]["filename"].apply(lambda x: get_string_segment(x, 6, '_'))
            content.loc[content["level_type"] == "model", ["level"]] = \
                    content[content.level_type == "model"]["filename"].apply(lambda x: get_string_segment(x, 6, '_'))
            content.loc[content["level_type"] == "soil", ["level"]] = \
                    content[content.level_type == "soil"]["filename"].apply(lambda x: get_string_segment(x, 6, '_'))

            content["level"] = content["level"].astype(int)
            content.to_pickle(self.content_pkl)

            return content

        # Downloaded database file and pickle file.
        self.content_log_path = os.path.join(DWDContent.cache_path, "content_nwp.log.bz2")
        self.content_pkl = os.path.join(DWDContent.cache_path, "opendata_dwd_content_nwp.pkl")
        # Lock to avoid simultaneous writing to contents.
        self.lock_file = os.path.join(DWDContent.cache_path, "content_nwp_lock")
        
        # Wait until the lock file does not exist anymore. If the file exists, the database is currently
        # updated by another enstools instance. As soon the lock file does not exist anymore,
        # we create the file and therefore acquire the lock ourselves.
        logging.info("Acquiring content file lock...")
        wait_time = 0.1
        while os.path.exists(self.lock_file):
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 10) # Exponential backoff
        
        # Locking file.
        f = open(self.lock_file, 'w')
        f.close()

        # Put this in try block so we release the lock also if an error occurs.
        try:
            logging.info("File lock acquired.")
            logging.info("Comparing hashes of current database and server database...")

            # Check if checksum of file at "https://opendata.dwd.de/weather/nwp/content.log.bz2"
            # matches checksum of local file. If not, redownload.
            if os.path.exists(self.content_log_path):
                with open(self.content_log_path, 'rb') as fh:
                    local_md5 = hashlib.md5(fh.read()).hexdigest()
                logging.info(f"Local hash: {local_md5}")
            else:
                local_md5 = 'missing'
                logging.info("Local content.log.bz2 does not exist.")

            # Download checksum of server file.
            logging.info("Downloading checksum of server file...")
            r = requests.get("https://opendata.dwd.de/weather/nwp/content.log.bz2")
            remote_md5 = hashlib.md5(r.content).hexdigest()
            logging.info(f"Remote hash: {remote_md5}")

            if local_md5 != remote_md5:
                refresh_content = True
                logging.info("Non-matching MD5 checksums for database, refreshing database.")

            if os.path.exists(self.content_pkl) and not refresh_content:
                logging.info(f"Reading content database from {self.content_pkl}")
                content_old = pandas.read_pickle(self.content_pkl)
                self.content = content_old

            else:
                if refresh_content:
                    logging.info("Refreshing content database")
                else:
                    logging.info("Initializing content database")
                if os.path.exists(self.content_log_path):
                    os.remove(self.content_log_path)

                download("https://opendata.dwd.de/weather/nwp/content.log.bz2",
                        destination=self.content_log_path, uncompress=False)
                self.content = create_dataframe(self.content_log_path)

            # Finished updating database. Deleting the lock file.
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
                
        except Exception as e:
            logging.error(e)
            # Delete lock file if something went wrong and we abort.
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
            raise e
            

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
            if len(avail_grid_types) == 1:
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
            if len(avail_level_types) == 1:
                level_type = avail_level_types[0]

            else:
                raise ValueError("You have to choose one of the level_types: {}".format(avail_level_types))
        else:
            if level_type not in avail_level_types and len(avail_level_types) > 0:
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
        avail_abs_init_times = content[(content["model"] == model)
                                       & (content["grid_type"] == grid_type)][
            "abs_init_time"].drop_duplicates().values.tolist()
        avail_abs_init_times.sort()
        for one_abs in avail_abs_init_times:
            avail_init_times.append(datetime.strptime(one_abs, "%Y%m%d%H"))
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
                             & self.__init_time_selection(init_time)]["variable"].drop_duplicates().values.tolist()
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
                                    & self.__init_time_selection(init_time)
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
                                       & self.__init_time_selection(init_time)
                                       & (content["variable"] == variable)
                                       & (content["level_type"] == level_type)]["forecast_hour"] \
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
                               & self.__init_time_selection(init_time)
                               & (content["variable"] == variable)
                               & (content["level_type"] == level_type)]["level"].drop_duplicates().values.tolist()
        avail_levels.sort()
        return avail_levels

    def __init_time_selection(self, init_time, content=None):
        """
        this function is internally used for indexing and selecting a specific init time.

        Parameters
        ----------
        init_time:  int or datetime or list of int or datetime
                    if an integer is given, the column init_time is used. For da datetime object
                    the column abs_init_time is used.
                    
        content:    pandas.DataFrame
                    Reference to the content to select from. If None, self.content is chosen.
                    Can also be a subset of the content for more efficient parsing.
        Returns
        -------
        Series:
                    pandas series object used for indexing.
        """
        if content is None:
            content = self.content
        
        if type(init_time) == int:
            return content["init_time"] == init_time
        elif isinstance(init_time, datetime):
            return content["abs_init_time"] == init_time.strftime("%Y%m%d%H")
        
        elif isinstance(init_time, list):
            if len(init_time) == 0:
                return None
            
            if type(init_time[0]) == int:
                return content["init_time"].isin(init_time)
            elif isinstance(init_time[0], datetime):
                return content["abs_init_time"].isin([it.strftime("%Y%m%d%H") for it in init_time])
            
            raise ValueError(f"invalid data type for init_time: {type(init_time)}")

    def __get_uniq_content_line(self, model=None, grid_type=None, init_time=None, variable=None,
                                level_type=None, forecast_hour=None, level=None, refreshed=False):
        """
        Use the given parameters to find ONE entry of the content.

        Parameters
        ----------
        model: str
            The model of the file to subset.
        grid_type: str
            The type of the geo grid.
        init_time: int
            The initialization time of the forecast of the file to subset.
        variable: str
            The variable of the file to subset.
        level_type: str
            The type of level of the file to subset.
        forecast_hour: int
            The hours since the initialization of the forecast of the file to subset.
        level: int
            The level of the file to subset.
        refreshed: bool
            Shows that the content database has just been updated before the method call.

        Returns
        -------
        pandas.DataFrame
        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content
        result = content[(content["model"] == model)
                         & self.__init_time_selection(init_time)
                         & (content["variable"] == variable)
                         & (content["level_type"] == level_type)
                         & (content["forecast_hour"] == forecast_hour)
                         & (content["level"] == level)
                         & (content["grid_type"] == grid_type)]

        # raise exception if nothing was found.
        if len(result) == 0:
            raise KeyError("No entry not found in content for (model: {}, grid_type: {}, init_time: {}, variable: {}, "
                           "level_type: {}, forecast_hour: {}, level: {})"
                           .format(model, grid_type, init_time, variable, level_type, forecast_hour, level))
        elif len(result) > 1:
            if refreshed:
                # This happens when the database has just been updated before the method call
                logging.warning("{} entries found in content, only one expected for (model: {}, grid_type: {}, "
                                "init_time: {}, variable: {}, level_type: {}, forecast_hour: {}, level: {}), "
                                "taking the newest!"
                                .format(len(result), model, grid_type, init_time, variable, level_type, forecast_hour,
                                        level))
                result = result.sort_values(by="time", ascending=False).iloc[[0]]
            else:
                # If there are duplicated entries found then update the database first and try it again:
                logging.warning("There have been some duplicated entries found, trying again with refreshed database.")
                self.refresh_content()
                return self.__get_uniq_content_line(model=model, grid_type=grid_type, init_time=init_time, variable=variable,
                                                    level_type=level_type, forecast_hour=forecast_hour, level=level, refreshed=True)
        return result
    
    
    
    def __get_content_block(self, model=None, grid_type=None, level_type=None, init_times=None, 
                            variables=None,  forecast_hours=None, levels=None, refreshed=False):
        """
        Given the parameters, return a list of contents from the content file. Call this instead of 
        @ref __get_uniq_content_line when you need entire blocks, this is much faster.
        
        Parameters
        ----------
        model: str
            The model of the file to subset.
        grid_type: str
            The type of the geo grid.
        init_times: list(int) or list(datetime)
            The initialization times of the forecast to subset.
        variables: list(str)
            The variables of the file to subset.
        level_type: str
            The type of level of the file to subset.
        forecast_hours: list(int)
            The hours since the initialization of the forecast of the file to subset.
        levels: list(int)
            The levels of the file to subset.
        refreshed: bool
            Shows that the content database has just been updated before the method call.

        Returns
        -------
        pandas.DataFrame
        """
        
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        content = self.content
        subset_columns = ["model", "level_type", "grid_type"]
        data_subset = content[(content["model"] == model)
                         & (content["level_type"] == level_type)
                         & (content["grid_type"] == grid_type)]
        
        if variables is not None:
            data_subset = data_subset[data_subset["variable"].isin(variables)]
            subset_columns.append("variable")
        if forecast_hours is not None:
            data_subset = data_subset[data_subset["forecast_hour"].isin(forecast_hours)]
            subset_columns.append("forecast_hour")
        if levels is not None:
            data_subset = data_subset[data_subset["level"].isin(levels)]
            subset_columns.append("level")
        if init_times is not None:
            data_subset = data_subset[self.__init_time_selection(init_times, content=data_subset)]
            if (type(init_times) == int) or (isinstance(init_times, list) and type(init_times[0]) == int):
                subset_columns.append("init_time")
            elif (isinstance(init_times, datetime)) or (isinstance(init_times, list) and isinstance(init_times[0], datetime)):
                subset_columns.append("abs_init_time")
        
        # Check whether the subset actually contains all possible combinations of the given input fields
        count_rows = data_subset.shape[0]
        
        # Drop duplicates and count them
        dropped_dups = data_subset[subset_columns].drop_duplicates()
        dropped_dups_count = dropped_dups.shape[0]
        
        # Get expected number of rows as product of number of unique elements along all dimensions
        uniques = data_subset[subset_columns].nunique()
        expected_rows = uniques.prod() 
        
        if dropped_dups_count < count_rows:
            if refreshed:
                # There are some rows duplicated. Log them and raise an error.
                dup_rows = data_subset[data_subset[subset_columns].duplicated(keep=False)]
                logging.warning(f"Duplicated entries found in cached content block. Duplicated data:\n {dup_rows}")
                return data_subset[~data_subset[subset_columns].duplicated(keep='first')] # return only first occurrences
            else:
                logging.warning("There have been some duplicated entries found, trying again with refreshed database.")
                self.refresh_content()
                return self.__get_content_block(model=model, grid_type=grid_type, level_type=level_type, init_times=init_times,
                                                variables=variables, forecast_hours=forecast_hours, levels=levels, refreshed=True)
        
        elif expected_rows > count_rows:
            # There are some rows missing, we couldn't find all combinations.
            # Extract actually which ones are missing and log them.
            unique_vals_per_col = [data_subset[col].unique() for col in subset_columns]
            # Use multiindex to create all combinations and see which of them are missing in our data frame
            multiindex = pandas.MultiIndex.from_product(
                    unique_vals_per_col,
                    names=subset_columns
            )
            data_subset_multi_index = (
                    data_subset
                    .set_index(subset_columns)
                    .reindex(multiindex, fill_value = 0)
                    .reset_index()
            )
            # missing -> concat and drop duplicates
            missing_rows = pandas.concat([data_subset, data_subset_multi_index]).drop_duplicates(keep=False)
            raise KeyError(f"Missing entries for retrieve request in cached content block. Missing rows:\n {missing_rows}")
            
        return data_subset
        
        
        

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
        url = self.__get_uniq_content_line(model=model, grid_type=grid_type, init_time=init_time, variable=variable,
                                           level_type=level_type, forecast_hour=forecast_hour, level=level)
        return url["file"].values[0]

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
        url = self.__get_uniq_content_line(model=model, grid_type=grid_type, init_time=init_time, variable=variable,
                                           level_type=level_type, forecast_hour=forecast_hour, level=level)
        return url["filename"].values[0]

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
        url = self.__get_uniq_content_line(model=model, grid_type=grid_type, init_time=init_time, variable=variable,
                                           level_type=level_type, forecast_hour=forecast_hour, level=level)
        return url["size"].values[0]

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
                         level_type=None, forecast_hour=None, levels=None, check_urls=True):
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
        init_time: int, datetime, or list of int or datetime
            The initialization time of the file.
        variable: list or str
            The variable of the file.
        level_type: str
            The type of level of the file.
        forecast_hour: list or int
            The hours of the forecast since the initialization of the simulation.
        levels: list or int
            The levels.
        check_urls: bool
            Whether to ping all download URLs to check their availability. 
            Might take a while with high-dimensional data sets.

        Returns
        -------

        """
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        params_available = True
        if model not in self.get_models():
            params_available = False
            
        # make init times a list if it isn't
        init_times = init_time if isinstance(init_time, list) else [init_time]
        avail_init_times = self.get_avail_init_times(model=model, grid_type=grid_type)
        
        for init_time in init_times:
            if init_time not in avail_init_times:
                params_available = False
                break
            
        avail_vars = self.get_avail_vars(model=model, grid_type=grid_type, init_time=init_times)
        
        for var in variable:
            if var not in avail_vars:
                params_available = False
                break

            avail_level_types = self.get_avail_level_types(model=model, grid_type=grid_type,
                                                           init_time=init_times, variable=var)
            if level_type not in avail_level_types:
                params_available = False
                break
            
            avail_forecast_hours = self.get_avail_forecast_hours(model=model, grid_type=grid_type, init_time=init_times,
                                                             variable=var, level_type=level_type)
            avail_levels = self.get_avail_levels(model=model, grid_type=grid_type, init_time=init_times, variable=var, 
                                                             level_type=level_type)
            
            for hour in forecast_hour:
                if hour not in avail_forecast_hours:
                    params_available = False
                    break

            for lev in levels:
                if lev not in avail_levels:
                    params_available = False
                    break
                        
            if not params_available:
                break

        if not params_available:
            logging.warning("Parameters not available")
            
        elif check_urls or (init_times is not None and (isinstance(init_times, list) and init_times[0] == int)):
            # Ping URLs if flag is set. We need to check anyways if init time is only given as int, cant deduce actual 
            # timesteps from it. An init_time of "0" is in the cache, but the remote's "0" might correspond to a different day.
            logging.info("Pinging URLs for availability...")
            if (init_times is not None and type(init_times[0]) == int):
                 logging.info("Consider setting init_time to an actual datetime instead of int, then this step can be skipped.")
                    
            url_indices = itertools.product(forecast_hour, variable, levels)
            for hour, var, lev in url_indices:
                if not self.check_url_available(model=model, grid_type=grid_type, init_time=init_times,
                                                variable=var, level_type=level_type, forecast_hour=hour, level=lev):
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
            for init_time in init_times:
                if init_time not in avail_init_times:
                    raise ValueError("The initial time {} is not available. Possible Values: {}"
                                 .format(init_time, avail_init_times))

            for var in variable:
                avail_vars = self.get_avail_vars(model=model, grid_type=grid_type, init_time=init_time)
                if var not in avail_vars:
                    raise ValueError("The variable {} is not available for the {} model "
                                     "and the init_time {}. Available variables: {}"
                                     .format(var, model, init_time, avail_vars))

                avail_level_types = self.get_avail_level_types(model=model, grid_type=grid_type,
                                                               init_time=init_time, variable=var)
                if level_type not in avail_level_types:
                    raise ValueError("The level type {} is not available for the variable {}. Available types: {}"
                                     .format(level_type, var, avail_level_types))
                avail_forecast_hours = self.get_avail_forecast_hours(model=model, grid_type=grid_type,
                                                                     init_time=init_time, variable=var,
                                                                     level_type=level_type)
                avail_levels = self.get_avail_levels(model=model, grid_type=grid_type, init_time=init_time,
                                                             variable=var, level_type=level_type)
                for hour in forecast_hour:
                    if hour not in avail_forecast_hours:
                        raise ValueError(
                            "The forecast hour {} is not available for the variable {}. Possible values: {}"
                                .format(hour, var, avail_forecast_hours))
                        
                for lev in levels:
                    if lev not in avail_levels:
                        raise ValueError("The level {} is not available for the variable {} and the level type {}."
                                             .format(lev, var, level_type)
                                             + " Possible Values: {}".format(avail_levels))

    def get_merge_dataset_name(self, model=None, variable=None,
                               level_type=None, init_time=None, forecast_hour=None):
        # find the absolute init time for the given init_time, use only the first variable
        abs_init_time = self.content[(self.content["model"] == model) &
                                     (self.content["level_type"] == level_type) &
                                     (self.content["variable"] == variable[0]) &
                                     self.__init_time_selection(init_time)]["abs_init_time"].drop_duplicates().values[0]

        # construct the actual file name
        merge_name = model + "_" + level_type + "_" + "init" + abs_init_time
        # add forecast time
        # do we have regular spaced data?
        if len(forecast_hour) == forecast_hour[-1] - forecast_hour[0] + 1:
            merge_name += f"+{forecast_hour[0]}h-{forecast_hour[-1]}h"
        else:
            for one_time in forecast_hour:
                merge_name += "+{}h".format(one_time)
        merge_name += "_"
        # add variable names
        for var in variable:
            merge_name = merge_name + var + "+"
        merge_name = merge_name[:-1]

        return merge_name + ".grib2"

    def retrieve(self, service="DWD", model="ICON", eps=None, grid_type=None, variable=None, level_type=None,
                 levels=0, init_time=None, forecast_hour=None, merge_files=False, dest=None, validate_urls=True):
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
        
        validate_urls : bool
                Whether to ping all download URLs first before starting to download the data. 
                Updates the cache if some URLs are not there. Considering setting this flag to 
                False for faster downloads, especially if you download sizeable amounts.
        
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
            
        grid_type = self.check_grid_type(model=model, grid_type=grid_type)
        level_type = self.check_level_type(model=model, grid_type=grid_type, init_time=init_time,
                                           variable=variable, level_type=level_type)
        
        logging.info("Checking parameter consistency with data source...")
        self.check_parameters(model=model, grid_type=grid_type, init_time=init_time, variable=variable,
                              level_type=level_type, forecast_hour=forecast_hour, levels=levels, check_urls=validate_urls)
        
        if merge_files:
            merge_dataset_name = dest + "/" + self.get_merge_dataset_name(model=model, variable=variable,
                                                                          level_type=level_type, init_time=init_time,
                                                                          forecast_hour=forecast_hour)
            if os.path.exists(merge_dataset_name):
                logging.warning("file not downloaded because it is already present: " + merge_dataset_name)
                return [merge_dataset_name]
        total_size = 0
        
        content_entries = self.__get_content_block(model=model, grid_type=grid_type, level_type=level_type,
                                 init_times=init_time, variables=variable,  forecast_hours=forecast_hour, levels=levels)
        
        download_urls = content_entries["file"].to_list()
        download_files = content_entries["filename"].to_list()
        total_size = content_entries["size"].sum() 
        
        download_files = [dest + "/" + file[:-4] for file in download_files]

        total_size_human = bytes2human(total_size)

        logging.info("Downloading {} files with the total size of {}".format(len(download_files), total_size_human))
        for i in range(len(download_urls)):
            download(download_urls[i], download_files[i] + ".bz2", uncompress=True)

        if merge_files:
            concat(download_files, merge_dataset_name)
            for file in download_files:
                os.remove(file)
            return [merge_dataset_name]
        else:
            return download_files


# store the created content object for later re-use
__content = None


def getDWDContent():
    """
    Creates a DWDContent object and returns it. Multiple calls to this function will return the same object.

    Returns
    -------
    DWDContent
    """
    global __content
    if __content is not None:
        return __content
    else:
        __content = DWDContent()
        return __content
