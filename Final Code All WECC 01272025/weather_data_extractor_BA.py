# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:48:54 2024

@author: Saleh
"""

# weather_data_extractor.py

import OpenVisus as ov
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store data that has already been loaded
loaded_datasets: dict[str, any] = {}
loaded_timesteps: dict[tuple[str, int], any] = {}


def get_weather_data_BA(year, month, day, variable_names, latitude, longitude, variables, model="ACCESS-CM2", scenarios=("historical", "ssp585"), lat_start=-59.875, lon_start=0.125, lat_step=0.25, lon_step=0.25):
    # Calculate the day of the year, accounting for leap years
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_month[1] = 29
    day_of_the_year = sum(days_in_month[:month - 1]) + day
    
    # Determine the appropriate scenario based on the year
    scenario = scenarios[0] if year < 2015 else scenarios[1]

    # Calculate the timestep
    timestep = year * 365 + day_of_the_year
    logger.info(f"Calculated timestep: {timestep}")

    def latlon_to_indices(lat, lon):
        lat_index = round((lat - lat_start) / lat_step)
        lon_index = round((lon - lon_start) / lon_step)
        return lat_index, lon_index

    Weather_data = {}

    for variable, var_name in zip(variables, variable_names):
        try:
            # Construct the dataset name
            dataset_name = f"{variable}_day_{model}_{scenario}_r1i1p1f1_gn"
            dataset_url = f"http://atlantis.sci.utah.edu/mod_visus?dataset={dataset_name}&cached=1"
            # logger.info(f"Loading dataset: {dataset_name}")

            # Open the dataset and store it if its the first time loading the dataset
            if dataset_url not in loaded_datasets:
                db = ov.LoadDataset(dataset_url)
                loaded_datasets[dataset_url] = db
            else:
                db = loaded_datasets[dataset_url]
            
            # Read a timestep from the dataset and store it
            url_timestep = (dataset_url, timestep)
            if url_timestep not in loaded_timesteps:
                data = db.read(time=timestep)
                loaded_timesteps[url_timestep] = data
            else:
                data = loaded_timesteps[url_timestep]
 
            # Convert latitude and longitude to array indices
            lat_index, lon_index = latlon_to_indices(latitude, longitude)

            # Extract the data for the specified latitude and longitude
            specific_data = data[lat_index, lon_index]
            
            # Convert temperatures from Kelvin to Celsius
            if variable in ["tas", "tasmax", "tasmin"]:
                specific_data -= 273.15
            
            # Store the data in the Weather_data dictionary with descriptive name
            Weather_data[variable] = {
                "description": var_name,
                "value": specific_data
            }
        except Exception as e:
            # logger.error(f"Error processing variable {variable}: {e}")
            Weather_data[variable] = {
                "description": var_name,
                "value": None
            }

    return Weather_data

# Example usage:
# weather_data = get_weather_data(2023, 6, 12, -33.865143, 151.209900)
# print(weather_data)
