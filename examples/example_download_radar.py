#!/usr/bin/env python3

from enstools.opendata import retrieve_radar, getDWDRadar

# search for data using the content object
content = getDWDRadar()

print("Available Product Classes:", content.get_product_classes())
print("Available products:", content.get_products())
data_times_re = content.get_avail_data_times(product="re")
print("Available data times for product 're' :", data_times_re)
forecast_times_re = content.get_avail_forecast_times(product="re", data_time=data_times_re[-1])
print("Available forecast times for youngest 're' data:", forecast_times_re)


data_times_pg = content.get_avail_data_times(product="pg")
print("Available data times for product 'pg' :", data_times_pg)
forecast_times_pg = content.get_avail_forecast_times(product="pg", data_time=data_times_pg[-1])
print("Available forecast times for youngest 'pg' data:", forecast_times_pg)
file_formats_pg = content.get_avail_file_formats(product="pg")
print("Available file formats for 'pg':", file_formats_pg)
# examples for data download
retrieve_radar(product="re", data_time=data_times_re[-1], forecast_time=forecast_times_re, dest="data")

retrieve_radar(product="pg", data_time=data_times_pg,
               file_format=file_formats_pg[0], dest="data")
