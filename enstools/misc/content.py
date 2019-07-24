import pandas
import numpy as np

a = pandas.read_csv("content.log",delimiter="|", header=None,names=["file", "size", "time"])
a = a[~a.file.str.contains("snow4")]
a = a[~a.file.str.contains("content.log")]
size = a["size"]
time = a["time"]
a["model"] = a["file"].apply(lambda x: x.split("/")[1])
a["file_type"] = a["file"].apply(lambda x: x.split("/")[2])
a["init_time"] = a["file"].apply(lambda x: x.split("/")[3])
a["variable"] = a["file"].apply(lambda x: x.split("/")[4])
a["filename"] = a["file"].apply(lambda x: x.split("/")[5])
a["file"] = a["file"].apply(lambda x: "https://opendata.dwd.de/weather/nwp" +x[1:])
a["level_type"] = a["filename"].apply(lambda x: x.split("_")[3])
a["forecast_time"] = a["filename"].apply(lambda x: x.split("_")[5])
a["levels"] = 0
a.loc[a["level_type"] == "singe-level",["levels"]] = 0
a.loc[a["level_type"] == "pressure-level",["levels"]] = a[a.level_type == "pressure-level"]["filename"].apply(lambda x: x.split("_")[6])
a.loc[a["level_type"] == "model-level",["levels"]] = a[a.level_type == "model-level"]["filename"].apply(lambda x: x.split("_")[6])



