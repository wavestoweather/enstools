from .DWDRadar import getDWDRadar


def retrieve_radar(product=None, data_time=None, forecast_time=0, dest=None, file_format=None, validate_urls=True):
    """
    Downloads radar datasets from opendata server.
    
    Parameters
    ----------
        product: str
            The type of the radar data.

        data_time : list or int
            Time of the radar data. Multiple values are allowed.

        forecast_time : list or int
                time since the initialization of forecast. Multiple values are allowed.

        dest : str
                Destination folder for downloaded data. If the files are already available,
                they are not downloaded again.
        file_format: str
            The file format (eg. 'gz'). Only has to be specified when there are more than one available.

        validate_urls : bool
                Whether to ping all download URLs first to validate state of the cache. Might slow
                down the process significantly for bigger data requests.
                
        Returns
        -------
        list :
                names of downloaded files.

    """
    content = getDWDRadar()
    download_files = content.retrieve(product=product, data_time=data_time, forecast_time=forecast_time,
                                      dest=dest, file_format=file_format, validate_urls=validate_urls)
    return download_files
