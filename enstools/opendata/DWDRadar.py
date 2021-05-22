import logging
import os
from pathlib import Path
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import urlopen
import pandas
from enstools.misc import download, bytes2human
from get_cache_dir import get_cache_dir


class DWDRadar:
    # create a new temporal directory to store content.log and opendata_dwd_content.pkl
    cache_path = get_cache_dir()

    def __init__(self, refresh_content=False):

        def create_dataframe(logdata):
            """
            Creates a pandas.DataFrame from the given log data from https://opendata.dwd.de/weather/radar/
            and sets the following attributes:
                file: str
                    Name of the url of the file.
                size: str
                    The size of the file.
                time: np.datetime64
                    The creation time on the server.
                product_class: str
                    The radar product_class, e.g. "radolan".
                format: str
                    The format of the file.
                filename: str
                    The name of the file.
                forecast_time: int
                    Minutes since the initalization of the forecast.
                    If the radar product is not a forecast this is alwys zero.

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
            content = pandas.read_csv(logdata, delimiter="|", header=None, names=["file", "size", "upload_time"])
            content["upload_time"] = pandas.to_datetime(content["upload_time"], format="%Y-%m-%d %H:%M:%S")
            content["size"] = content["size"].astype(int)

            # ignore the content.log file and tar.gz files
            content = content[~content.file.str.contains("content.log")]
            content = content[~content.file.str.contains(".tar.")]

            content["product_class"] = content["file"].apply(lambda x: x.split("/")[1])
            content["product"] = content["file"].apply(lambda x: x.split("/")[2])
            content = content[(content.product_class == "composit") |
                              (content.product_class == "radvor") |
                              (content.product_class == "radolan")]
            content["filename"] = content["file"].apply(lambda x: x.split("/")[3])
            content["data_time"] = content["filename"]
            content["data_time"] = content["data_time"].apply(lambda x: datetime.strptime(x[15:-10], "%y%m%d%H%M")
                                                              if x.endswith("-dwd---bin")
                                                              else datetime.strptime(x[2:-7], "%y%m%d%H%M")
                                                              if (x.startswith("RE") | x.startswith("RQ"))
                                                              else datetime.strptime(x[2:-8], "%y%m%d%H%M")
                                                              if (x.startswith("FX") | x.startswith("WN"))
                                                              else self.__scrape_pg_paah_time(x)
                                                              if x.startswith("PAAH")
                                                              else datetime.strptime(x[15:-11], "%y%m%d%H%M")
                                                              if x.endswith("-dwd---bufr")
                                                              else None)
            content["forecast_time"] = content["filename"]
            content["forecast_time"] = content["forecast_time"].apply(lambda x: int(x[-6:-3])
                                                                      if (x.startswith("RE")
                                                                          or x.startswith("RQ"))
                                                                      else int(x[-7:-4])
                                                                      if x.startswith("WN")
                                                                      else 0)
            content["format"] = content["filename"]
            content["format"] = content["format"].apply(lambda x:
                                                        "bin" if x.endswith("bin")
                                                        else "bufr" if x.endswith("bufr")
                                                        else "gz" if x.endswith("gz")
                                                        else "buf" if x.endswith(".buf")
                                                        else ".bz2" if x.endswith(".bz2")
                                                        else None)

            content["file"] = content["file"].apply(lambda x: "https://opendata.dwd.de/weather/radar" + x[1:])

            content.to_pickle(self.content_pkl_path)
            return content

        self.content_pkl_path = os.path.join(DWDRadar.cache_path, "opendata_dwd_content_radar.pkl")
        if os.path.exists(self.content_pkl_path) and not refresh_content:
            logging.info(f"Reading content database from {self.content_pkl_path}")
            content_old = pandas.read_pickle(self.content_pkl_path)
            self.content = content_old

        else:
            if refresh_content:
                logging.info("Refreshing content database")
            else:
                logging.info("Initializing content database")
            if os.path.exists(os.path.join(DWDRadar.cache_path, "content.log")):
                os.remove(os.path.join(DWDRadar.cache_path, "content.log"))
            self.content_log_path = os.path.join(DWDRadar.cache_path, "content_radar.log.bz2")
            download("https://opendata.dwd.de/weather/radar/content.log.bz2",
                     destination=self.content_log_path, uncompress=False)
            self.content = create_dataframe(self.content_log_path)

    @staticmethod
    def __scrape_pg_paah_time(filename):
        """
        Determine the missing year and month in PAAH filenames from todays date.
        """
        date_now = datetime.utcnow()
        file_date = datetime.strptime(filename[10:-4], "%d%H%M")

        # If somebody wants to download data after new year party:
        if date_now.month == 1 and date_now.day <= 3:
            if file_date.day >= 28:
                file_date = file_date.replace(year=date_now.year-1, month=12)
        else:
            file_date = file_date.replace(year=date_now.year, month=date_now.month)
        return file_date

    def refresh_content(self):
        """
        Initializes the DWDRadar object again.
        Downloads the actual content.log from the server and creates a new DataFrame.
        """
        self.__init__(refresh_content=True)

    def get_product_classes(self):
        """
        Gives the available product classes of the data server.
        Returns
        -------
            List of Strings with the available product classes

        """
        content = self.content
        avail_product_classes = content["product_class"].drop_duplicates().values.tolist()
        avail_product_classes.sort()
        return avail_product_classes

    def get_products(self, product_class=None):
        """
        Gives the available radar products.
        Parameters
        ----------
        product_class: string
            The group of the radar data products.

        Returns
        -------
        If available, the right data products will be returned.
        """
        content = self.content
        if product_class is None:
            avail_products = content["product"].drop_duplicates().values.tolist()
        else:
            avail_products = content[content["product_class"] == product_class]["product"]\
                .drop_duplicates().values.tolist()
        return avail_products

    def get_avail_data_times(self, product=None):
        """
        Gives the available data times for a given radar product.

        Parameters
        ----------
        product: str
            The name of the data product.

        Returns
        -------
        avail_data_times: list
            A list of integers of the available data times.

        """
        content = self.content
        avail_data_times = content[(content["product"] == product)]["data_time"].drop_duplicates().dt\
            .to_pydatetime().tolist()
        avail_data_times.sort()
        return avail_data_times

    def get_avail_forecast_times(self, product=None, data_time=None):
        """
        Gives the available  forecast times since the data_time for a given radar product and data_time.

        Parameters
        ----------
        product: str
            The name of the data product.
        data_time: int
            The time of the radar data for which the available forecast times want to be known.

        Returns
        -------
        avail_forecast_times: list
            The available times of the radar data since the initialization of the forecast.

        """
        content = self.content
        avail_forecast_times = content[(content["product"] == product)
                                       & (content["data_time"] == data_time)]["forecast_time"] \
            .drop_duplicates().values.tolist()
        avail_forecast_times.sort()
        return avail_forecast_times

    def get_avail_file_formats(self, product=None):
        """
        Gives the available file formats for a given radar product.
        Currently most products only have one format to choose.

        Parameters
        ----------
        product: str
            The name of the data product.

        Returns
        -------
        A list of strings of the available formats.

        """
        content = self.content
        avail_file_formats = content[content["product"] == product]["format"].drop_duplicates().values.tolist()
        avail_file_formats.sort()
        return avail_file_formats

    def get_url(self, product=None, data_time=None, forecast_time=None, file_format=None):
        """
        Gives the url of the file on the https://opendata.dwd.de/weather/radar server.

        Parameters
        ----------
        product: str
        data_time: int
            The time of the radar data file for which the url wants to be known.
        forecast_time: int
            The minutes since the data_time of the radar data of the file for which the url wants to be known.
        file_format: str
            The file format (eg. 'gz').

        Returns
        -------
        url: str
            The url adress of the file.
        """
        content = self.content

        url = content[(content["product"] == product)
                      & (content["data_time"] == data_time)
                      & (content["forecast_time"] == forecast_time)
                      & (content["format"] == file_format)]["file"].values
        if len(url) != 1:
            raise KeyError(f"Url not found(product:{product}, data_time:{data_time}, "
                           f"forecast_time:{forecast_time}, format:{file_format})")

        return url.item()

    def get_filename(self, product=None, data_time=None, forecast_time=None, file_format=None):
        """
         Gives the filename of the file on the https://opendata.dwd.de/weather/radar server.

        Parameters
        ----------
        product: str
            The name of the data product.
        data_time: datetime.datetime
            The time of the radar data file for which the filename wants to be known.
        forecast_time: int
            The minutes since the data_time of the radar data of the file for which the filename wants to be known.
        file_format: str
            The file format (eg. '.gz')
        Returns
        -------
        url: str
            The url adress of the file.
        """
        content = self.content

        filename = content[(content["product"] == product)
                           & (content["data_time"] == data_time)
                           & (content["forecast_time"] == forecast_time)
                           & (content["format"] == file_format)]["filename"].values.item()

        return filename

    def check_url_available(self, product=None, data_time=None, forecast_time=None, file_format=None):
        try:
            http_code = urlopen(self.get_url(product=product, data_time=data_time,
                                             forecast_time=forecast_time, file_format=file_format)).getcode()
            if http_code == 200:
                url_available = True
            else:
                url_available = False
        except HTTPError:
            url_available = False
        return url_available

    def get_file_size(self, product=None, data_time=None, forecast_time=None, file_format=None):
        content = self.content
        file_size = content[(content["product"] == product)
                            & (content["data_time"] == data_time)
                            & (content["forecast_time"] == forecast_time)
                            & (content["format"] == file_format)]["size"].values.item()
        return file_size

    def get_size_of_download(self, product=None, data_time=None, forecast_time=None, file_format=None):
        total_size = 0
        for dtime in data_time:
            for ftime in forecast_time:
                total_size += self.get_file_size(product=product, data_time=dtime,
                                                 forecast_time=ftime, file_format=file_format)
        return total_size

    def check_parameters(self, product=None, data_time=None, forecast_time=None, file_format=None):
        """
        Checks if there are all files available for the given parameters.
        If not, the DWDRadar object will be refreshed.
        If one file is not available for the given parameters, a detailed error will be thrown.

        Parameters
        ----------
        product: str
            The name of the data product.
        data_time: list
            The time of the radar data file.
        forecast_time: list
            The minutes since the data_time of the radar data.
        file_format: str
            The file format (eg. 'gz').

        Returns
        -------
        None
        """

        params_available = True

        if product not in self.get_products():
            params_available = False

        for dtime in data_time:
            avail_dtimes = self.get_avail_data_times(product=product)
            if dtime not in avail_dtimes:
                params_available = False
                break

            for ftime in forecast_time:
                if ftime not in self.get_avail_forecast_times(product=product, data_time=dtime):
                    params_available = False
                    break

        if not params_available:
            logging.warning("Parameters not available")
        else:
            for dtime in data_time:
                for ftime in forecast_time:
                    if not self.check_url_available(product=product, data_time=dtime,
                                                    forecast_time=ftime, file_format=file_format):
                        params_available = False
                        break
            if not params_available:
                logging.warning("Database outdated.")

        if not params_available:
            self.refresh_content()

            for dtime in data_time:
                avail_dtimes = self.get_avail_data_times(product=product)
                if dtime not in avail_dtimes:
                    raise ValueError(
                        "The data time {} is not available for the product {}. "
                        "Available variables:{}".format(dtime, product, avail_dtimes))

                avail_forecast_times = self.get_avail_forecast_times(product=product, data_time=dtime)
                for ftime in avail_forecast_times:

                    if ftime not in avail_forecast_times:
                        raise ValueError("The forecast time {} is not available for the data time {}. "
                                         "Possible values:{}".format(ftime, dtime, avail_forecast_times))

    def retrieve(self, product=None, data_time=None, forecast_time=0, dest=None, file_format=None):
        """
        Downloads datasets from opendata server.
        Parameters
        ----------
        product: str
            The type of the radar data.

        data_time : list or int
            Time of the radar data. Multiple values are allowed.

        forecast_time : list or int
                time since the initialization of forecast. Multiple values are allowed.

        file_format : str
            The file format (eg. 'gz'). Only has to be specified when there are more than one available.

        dest : str
                Destination folder for downloaded data. If the files are already available,
                they are not downloaded again.

        Returns
        -------
        list :
                names of downloaded files.
        """
        # Want to download data for one or more times?
        if not isinstance(data_time, (list, tuple)):
            data_time = [data_time]
        # Want to download one or more forecast times?
        if not isinstance(forecast_time, (list, tuple)):
            forecast_time = [forecast_time]

        if not os.path.exists(dest):
            os.mkdir(dest)
        if product is not None:
            product = product.lower()

        if file_format is None:
            avail_formats = self.get_avail_file_formats(product=product)
            if len(avail_formats) == 1:
                file_format = avail_formats[0]
            else:
                raise KeyError(f"The format for the product {product} is ambigous."
                               " You have to specify it in the 'format' keyword")

        download_files = []
        download_urls = []

        self.check_parameters(product=product, data_time=data_time,
                              forecast_time=forecast_time, file_format=file_format)
        for dtime in data_time:
            for ftime in forecast_time:
                download_urls.append(self.get_url(product=product, data_time=dtime,
                                                  forecast_time=ftime, file_format=file_format))
                download_files.append(self.get_filename(product=product, data_time=dtime,
                                                        forecast_time=ftime, file_format=file_format))

        download_files = [os.path.join(dest, file) for file in download_files]

        total_size_human = bytes2human(self.get_size_of_download(product=product,
                                                                 data_time=data_time,
                                                                 forecast_time=forecast_time,
                                                                 file_format=file_format))
        logging.info("Downloading {} files with the total size of {}".format(len(download_files), total_size_human))
        for i in range(len(download_urls)):
            download(download_urls[i], download_files[i], uncompress=False)

        return download_files


# store the created content object for later re-use
__content = None


def getDWDRadar():
    """
    Creates a DWDRadar object and returns it. Multiple calls to this function will return the same object.

    Returns
    -------
    DWDRadar
    """
    global __content
    if __content is not None:
        return __content
    else:
        __content = DWDRadar()
        return __content
