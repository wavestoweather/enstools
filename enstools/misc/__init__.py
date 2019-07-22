import os
import logging
import bz2
import six
import xarray
import numpy as np
from numba import jit
import dask
from enstools.io import read
from datetime import datetime, timedelta

if six.PY3:
    from urllib.request import urlretrieve, urlopen
else:
    from urllib import urlretrieve



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

    a1 : int
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