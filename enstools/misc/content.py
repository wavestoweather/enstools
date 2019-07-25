import pandas
import numpy as np


class DWD_Content():
    def __init__(self):


        content = pandas.read_csv("content.log", delimiter="|", header=None, names=["file", "size", "time"])
        content = content[~content.file.str.contains("snow4")]
        content = content[~content.file.str.contains("content.log")]

        content["model"] = content["file"].apply(lambda x: x.split("/")[1])
        content["file_type"] = content["file"].apply(lambda x: x.split("/")[2])
        content["init_time"] = content["file"].apply(lambda x: x.split("/")[3])
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
        content.loc[content["level_type"] == "pressure", ["level"]] = content[content.level_type == "pressure"]["filename"].apply(
            lambda x: x.split("_")[6])
        content.loc[content["level_type"] == "model", ["level"]] = content[content.level_type == "model"]["filename"].apply(
            lambda x: x.split("_")[6])
        content["level"] = content["level"].astype(int)

        self.content = content

    def get_models(self):
        content = self.content
        avail_models = content["models"].drop_duplicates().values.tolist()
        avail_models.sort()
        return avail_models

    def get_avail_init_times(self, model=None):
        content = self.content
        avail_init_times = content[content["model"] == model]["init_time"].drop_duplicates().values.tolist()
        avail_init_times.sort()
        return avail_init_times

    def get_avail_vars(self, model=None, init_time=None):
        content = self.content
        avail_vars =  content[(content["model"] == model)
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
                      & (content["forecast_hour"] == forecast_hour) & (content["level"] == level)]["file"].values[0]

        return url




a = DWD_Content()

print(a.get_avail_init_times(model="icon"))

print(a.get_avail_vars(model="icon", init_time="00"))

print(a.get_avail_level_types(model="icon", init_time="00", variable="t"))

print(a.get_avail_forecast_hours(model="icon", init_time="00", variable="t", level_type="pressure"))

print(a.get_avail_levels(model="icon", init_time="00", variable="t", level_type="pressure"))

print(a.get_url(model="icon", init_time="00", variable="t", level_type="pressure", forecast_hour=0, level=30))








# get variables of self init_time for self specific model:
# self[self.model == "icon"][self.init_time == "00"]["variable"].drop_duplicates().values.tolist()
# class content
# init oben
# 

