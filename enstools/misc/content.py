import pandas
import numpy as np

a = pandas.read_csv("content.log",delimiter="|", header=None,names=["file", "size", "time"])
a = a[~a.file.str.contains("snow4")]
a = a[~a.file.str.contains("content.log")]

a["model"] = a["file"].apply(lambda x: x.split("/")[1])
a["file_type"] = a["file"].apply(lambda x: x.split("/")[2])
a["init_time"] = a["file"].apply(lambda x: x.split("/")[3])
a["variable"] = a["file"].apply(lambda x: x.split("/")[4])
a["filename"] = a["file"].apply(lambda x: x.split("/")[5])
a["file"] = a["file"].apply(lambda x: "https://opendata.dwd.de/weather/nwp" +x[1:])

a["level_type"] = a["filename"].apply(lambda x: x.split("_")[3])
a["level_type"] = a["level_type"].apply(lambda x: x[:-6])

a["forecast_time"] = a["filename"].apply(lambda x: x.split("_")[5])
a.loc[a["level_type"] == "time-inv",["forecast_time"]] = "0"
a["forecast_time"] = a["forecast_time"].astype(int)

a["levels"] = "0"
a.loc[a["level_type"] == "single",["levels"]] = "0"
a.loc[a["level_type"] == "pressure",["levels"]] = a[a.level_type == "pressure"]["filename"].apply(lambda x: x.split("_")[6])
a.loc[a["level_type"] == "model",["levels"]] = a[a.level_type == "model"]["filename"].apply(lambda x: x.split("_")[6])
a["levels"] = a["levels"].astype(int)



# get variables of a init_time for a specific model:
# a[a.model == "icon"][a.init_time == "00"]["variable"].drop_duplicates().values.tolist()
# class content
# init oben
# 

