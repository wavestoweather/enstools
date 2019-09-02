from .DWDContent import getDWDContent


def retrieve(service="DWD", model="ICON", eps=None, grid_type=None, variable=None, level_type=None,
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
    content = getDWDContent()
    download_files = content.retrieve(service=service, model=model, eps=eps, grid_type=grid_type,
                                               variable=variable, level_type=level_type, levels=levels,
                                               init_time=init_time, forecast_hour=forecast_hour,
                                               merge_files=merge_files, dest=dest)
    return download_files
