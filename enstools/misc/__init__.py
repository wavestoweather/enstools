import os
import logging
import bz2
import six
import xarray
import numpy as np
from numba import jit
import dask
import pandas
from datetime import datetime
from shutil import copyfileobj


if six.PY3:
    from urllib.request import urlretrieve, urlopen
    from urllib.error import HTTPError
else:
    from urllib import urlretrieve


def download(url, destination, uncompress=True):
    """
    Download a file from the web

    Parameters
    ----------
    url : string
            the web address

    destination : string
            path to store the file. The file will only be downloaded once.

    uncompress : bool
            if True and if the name ends with '.bz2', it will
            automatically be uncompressed.
    """
    if destination.endswith(".bz2") and uncompress:
        destination_intern = destination[:-4]
    else:
        destination_intern = destination

    if os.path.exists(destination_intern):
        logging.warning("file not downloaded because it is already present: %s" % url)
        return

    # download
    logging.info("downloading %s ..." % os.path.basename(destination))
    logging.debug("from: %s" % url)
    fn, hd = urlretrieve(url, destination)

    # uncompress
    if destination.endswith(".bz2") and uncompress:
        # uncompress
        bfile = bz2.BZ2File(destination)
        dfile = open(destination_intern, "wb")
        dfile.write(bfile.read())
        dfile.close()
        bfile.close()
        # delete compressed
        os.remove(destination)


@jit(["b1(f4[:],f4[:],f4,f4)", "b1(f8[:],f8[:],f8,f8)"], nopython=True)
def point_in_polygon(polyx, polyy, testx, testy):
    """
    check whether or not a given coordinate is inside or outside of a polygon

    Parameters
    ----------
    polyx : np.ndarray
            x-coordinates of the polygon

    polyy : np.ndarray
            y-coordinates of the polygon

    testx : float
            x-coordinate of the point

    testy : float
            y-coordinate of the point

    Returns
    -------
    bool
            True, if the point is inside of the polygon
    """
    res = False
    j = polyx.shape[0] - 1
    for i in range(polyx.shape[0]):
        if ((polyy[i] > testy) != (polyy[j] > testy)) \
                and (testx < (polyx[j] - polyx[i]) * (testy - polyy[i]) / (polyy[j] - polyy[i]) + polyx[i]):
            res = not res
        j = i
    return res


def generate_coordinates(res, grid="regular"):
    """
    Generate grid coordinates for different types of grids. Currently only regular grids are implemented.

    Parameters
    ----------
    res : float
            resolution of the grid to generate

    grid : {'regular'}
            type of grid to generate. Currently only regular grids are supported

    Returns
    -------
    lon, lat : xarray.DataArray
            tuple of coordinate arrays

    Examples
    --------
    >>> lon, lat = generate_coordinates(20.0, "regular")
    >>> lon
    <xarray.DataArray 'lon' (lon: 18)>
    array([-180., -160., -140., -120., -100.,  -80.,  -60.,  -40.,  -20.,    0.,
             20.,   40.,   60.,   80.,  100.,  120.,  140.,  160.])
    Dimensions without coordinates: lon
    Attributes:
        units:    degrees_east
    >>> lat
    <xarray.DataArray 'lat' (lat: 9)>
    array([-80., -60., -40., -20.,   0.,  20.,  40.,  60.,  80.])
    Dimensions without coordinates: lat
    Attributes:
        units:    degrees_north
    """
    if grid == "regular":
        lon = xarray.DataArray(np.arange(-180, 180, res), dims=("lon",), name="lon", attrs={"units": "degrees_east"})
        lat = xarray.DataArray(np.arange(-90+res/2.0, 90, res), dims=("lat",), name="lat", attrs={"units": "degrees_north"})
    else:
        raise ValueError("unsupported grid type: '%s'" % grid)

    return lon, lat


def swapaxis(array, a1, a2):
    """
    move an axis of an array from one position to another. This function belongs to the numpy library but is not
    implemented for xarray.DataArray

    Parameters
    ----------
    array : xarray.DataArray or np.ndarray
            array to manipulate

    a1 : int
            first axis

    a2 : int
            second axis

    Returns
    -------
    xarray.DataArray or np.ndarray
            array with swap axis
    """
    dims = np.arange(array.ndim)
    dims[a1] = a2
    dims[a2] = a1
    dims = tuple(dims)
    if isinstance(array, xarray.DataArray):
        dims = tuple(map(lambda x:array.dims[x], dims))
    return array.transpose(*dims)


def has_ensemble_dim(ds):
    """
    check whether or not a dataset or xarray has already an ensemble dimension

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    bool
    """
    return get_ensemble_dim(ds) is not None


def get_ensemble_dim(ds):
    """
    get the name of the ensemble dimension from a dataset or array

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str or None :
            if no ensemble dimension was found, None is returned.
    """
    ens_names = ["ens", "ensemble", "member", "members"]
    for ens_name in ens_names:
        if ens_name in ds.dims:
            logging.debug("get_ensemble_dim: found name '%s'" % ens_name)
            return ens_name
    return None


def set_ensemble_member(ds, member):
    """
    set the number of the ensemble member. The Dataset is modified inplace.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
            Array or Dataset with ensemble dimension and only one member
    """
    # get the name of the ensemble dimension
    ens_dim = get_ensemble_dim(ds)

    # if we dont have one, add one
    if ens_dim is None:
        add_ensemble_dim(ds, member)
    else:
        if ds.coords[ens_dim].size == 1:
            ds.coords[ens_dim] = [member]
        else:
            logging.debug("set_ensemble_member: ds has more than one member. Doing nothing!")


def get_time_dim(ds):
    """
    get the name of the time dimension from a dataset or array

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    str or None :
            if no ensemble dimension was found, None is returned.
    """
    time_names = ["time", "Time", "times", "Times"]
    for time_name in time_names:
        if time_name in ds.dims:
            logging.debug("get_time_dim: found name '%s'" % time_name)
            return time_name
    return None


def has_dask_arrays(dataset):
    """
    check whether or not a dataset contains dask arrays

    Parameters
    ----------
    dataset : xarray.Dataset

    Returns
    -------
    bool :
            True if dask arrays are found
    """
    has_dask = False
    for varname, var in six.iteritems(dataset.variables):
        if isinstance(var.data, dask.array.core.Array):
            has_dask = True
            break
    return has_dask


def add_ensemble_dim(ds, member, inplace=True):
    """
    create an ensemble dimension with in the dataset

    Parameters
    ----------
    ds : xarray.Dataset

    member : int
            number of the ensemble member

    inplace : bool
            modify the dataset directly?

    Returns
    -------
    xarray.Dataset:
            A copy of the dataset expanded by the ensemble dimension or the expanded dataset
            if the expansion was done inplace.
    """
    # create a lazy copy of the dataset
    if inplace:
        new_ds = ds
    else:
        new_ds = ds.copy()

    # loop over all data variables.
    # those with time dimension are extended behind the time dimension, others at the front
    for one_name, one_var in six.iteritems(ds.data_vars):
        if is_additional_coordinate_variable(one_var):
            continue
        if len(one_var.dims) > 0 and one_var.dims[0] == "time":
            new_ds[one_name] = one_var.expand_dims("ens", 1)
        else:
            new_ds[one_name] = one_var.expand_dims("ens")
    new_ds.coords["ens"] = [member]

    # remove the ensemble_member attribute if present. it if only intented for datasets without ensemble dimension
    if "ensemble_member" in new_ds.attrs:
        del new_ds.attrs["ensemble_member"]
    return new_ds


def is_additional_coordinate_variable(var):
    """
    check if a variable belongs to a list of known constant coordinate variables

    Parameters
    ----------
    var : xarray.DataArray

    Returns
    -------
    bool :
           True = variable is not different for ensemble members
    """

    # list of excluded variables from different models
    excluded = {
        # COSMO variables
        "time_bnds": ("time", "bnds"),
        "slonu": ("rlat", "srlon"),
        "slatu": ("rlat", "srlon"),
        "slonv": ("srlat", "rlon"),
        "slatv": ("srlat", "rlon"),
        "vcoord": ("level1",),
        "soil1_bnds": ("soil1", "bnds"),
        "rotated_pole": (),
        "height_2m": (),
        "height_10m": (),
        "height_toa": (),
        "wbt_13c": (),
    }

    # variable is in excluded list?
    if var.name in excluded and excluded[var.name] == var.dims:
        return True
    else:
        return False


def first_element(array):
    """
    returns the first element of an array or the value of an scalar.

    Parameters
    ----------
    array : xarray.DataArray
            array with 0 or more dimensions.

    Returns
    -------
    float :
            first value.
    """
    if array.size > 1:
        return float(array[0])
    else:
        return float(array)


def count_ge(array, th=0):
    """
    count the number of values above a given threshold (>=)
    Parameters
    ----------
    array : xarray.DataArray and numpy.ndarray
            an array with arbitrary number of dimensions

    th : float
            threshold to test the array for.

    Returns
    -------
    int :
            number of values greater then or equal the threshold.
    """
    if type(array) == xarray.DataArray:
        return __count_ge(array.data, th)
    else:
        return __count_ge(array, th)


@jit(nopython=True)
def __count_ge(array, th):
    """
    numba implementation of count_ge
    """
    result = 0
    for i in range(array.size):
        if array.flat[i] >= th:
            result += 1
    return result


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
            merge_dataset_name = dest + "/" + service + "_" + model + "_" \
                                 + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%S%fs") + ".grib2"

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
    with open(out_filename, "wb") as out:
        for filename in files:
            with open(filename, "rb") as file:
                copyfileobj(file, out)


def bytes2human(n, format='%(value).1f %(symbol)s', symbols='iec'):
    """
    Convert n bytes into a human readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs
    """
    SYMBOLS = {
        'customary': ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'),
        'customary_ext': ('byte', 'kilo', 'mega', 'giga', 'tera', 'peta', 'exa', 'zetta', 'iotta'),
        'customary_with_B': ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'),
        'iec': ('Bi', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'),
        'iec_ext': ('byte', 'kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi', 'zebi', 'yobi'),
        'iec_with_B': ('Bi', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'),
    }

    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    if "iec" in symbols:
        base = 1024
    else:
        base = 1000
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = base ** i
    for symbol in reversed(symbols):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


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
                                               variable=variable, level_type=level_type, levels=levels, init_time=init_time,
                                               forecast_hour=forecast_hour, merge_files=merge_files, dest=dest)
    return download_files
