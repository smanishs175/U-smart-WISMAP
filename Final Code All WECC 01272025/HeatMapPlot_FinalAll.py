"""
Created on Tue Jun 15 10:21:34 2024

@author: Saleh
"""


# plot_heatmap.py


import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import OpenVisus as ov
import math
import os
from matplotlib.colors import LinearSegmentedColormap
from datetime import date, timedelta


# Store data that has already been loaded
loaded_datasets: dict[str, any] = {}
loaded_timesteps: dict[tuple[str, int], any] = {}

def plot_HeatMap(year, month, day, min_lat, max_lat, min_lon, max_lon, heatmap_index):
    # Enable to reload images even if it already exists on disk
    debug = False

    # Mapping heatmap_index to heatmap_index_des
    heatmap_descriptions = {
        0: 'MaxTemp',
        4: 'AvgTemp',
        8: 'MinTemp',
    }
    
    # Get the description based on the heatmap_index
    heatmap_index_des = heatmap_descriptions.get(heatmap_index, 'Unknown')
    
    
    if heatmap_index == 0:
        variables = ("tasmax",)
    elif heatmap_index == 4:
        variables = ("tas",)
    elif heatmap_index == 8:
        variables = ("tasmin",)
    else:
        print("Heatmap index is not defined")

    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Constants for lat/lon to index conversion
    LAT_START = -59.875
    LON_START = 0.125
    LAT_STEP = 0.25
    LON_STEP = 0.25

    image_cache_dir = os.path.join('static', 'heatmap_cache')

    if not os.path.isdir(image_cache_dir):
        os.mkdir(image_cache_dir)

    # Construct the image path
    heatmap_image_path = os.path.join(
        image_cache_dir, f'heatmap_image_{heatmap_index_des}_{year}_{month}_{day}.png'
    )
    
    # Function to calculate the day of the year, accounting for leap years
    def calculate_day_of_year(year, month, day):
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            days_in_month[1] = 29
        return sum(days_in_month[:month - 1]) + day

    # Function to convert latitude and longitude to array indices
    def latlon_to_indices(lat, lon, lat_start=LAT_START, lon_start=LON_START, lat_step=LAT_STEP, lon_step=LON_STEP):
        lat_index = round((lat - lat_start) / lat_step)
        lon_index = round((lon - lon_start) / lon_step)
        return lat_index, lon_index

    # Function to fetch weather data for given date, location, and parameters
    def fetch_weather_data(year, month, day, lat_range, lon_range, model="ACCESS-CM2", scenario="ssp585"):
        # Calculate the day of the year
        day_of_the_year = calculate_day_of_year(year, month, day)

        if heatmap_index == 0:
            variable_names = {
                "tasmax": "Daily Maximum Near-Surface Air Temperature"
            }
        elif heatmap_index == 4:
            variable_names = {
                "tas": "Daily Near-Surface Air Temperature"
            }
            
        elif heatmap_index == 8:
            variable_names = {
                "tasmin": "Daily Minimum Near-Surface Air Temperature"
            }
        else:
            print("Heatmap index is not defined")


        scenario = "historical" if year < 2015 else scenario
        timestep = year * 365 + day_of_the_year

        Weather_data = {}

        for variable in variable_names:
            try:
                dataset_name = f"{variable}_day_{model}_{scenario}_r1i1p1f1_gn"
                dataset_url = f"http://atlantis.sci.utah.edu/mod_visus?dataset={dataset_name}&cached=1"

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

                lat_indices = np.arange(round((lat_range[0] - LAT_START) / LAT_STEP), round((lat_range[1] - LAT_START) / LAT_STEP) + 1)
                lon_indices = np.arange(round((lon_range[0] - LON_START) / LON_STEP), round((lon_range[1] - LON_START) / LON_STEP) + 1)

                data_subset = data[lat_indices[:, None], lon_indices]

                if variable == "tasmax":  # Adjust temperature conversion logic here as needed
                    data_subset -= 273.15
                if variable == "tasmin":  # Adjust temperature conversion logic here as needed
                    data_subset -= 273.15
                if variable == "tas":  # Adjust temperature conversion logic here as needed
                    data_subset -= 273.15

                Weather_data[variable] = {
                    "description": variable_names[variable],
                    "value": data_subset
                }
            except Exception as e:
                logger.error(f"Error processing variable {variable}: {e}")
                Weather_data[variable] = {
                    "description": variable_names.get(variable, "Unknown"),
                    "value": None
                }

        return Weather_data

    # Function to plot weather data with Matplotlib
    def plot_weather_data(weather_data, variable, vmin, vmax, image_path):
        variable_data = weather_data[variable]["value"]

        # If there are any nan values, replace them with a low value
        variable_data = np.nan_to_num(variable_data, nan=-255)
        variable_data = np.flip(variable_data, 0)

        # Normalize the data between the provided min and max values
        norm = Normalize(vmin=vmin, vmax=vmax)
        normalized_data = norm(variable_data)

        # Define the custom colormap
        colors = [
            (0, 0, 1),    # blue
            (1, 1, 1),    # white
            (1, 1, 0),    # yellow
            (1, 0, 0),    # red
            (0, 0, 0)     # black
        ]
        
        n_bins = 100  # discretizes the interpolation into bins
        
        # Apply the colormap on the normalized data
        colormap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=n_bins)
        rgba_data = colormap(normalized_data)

        # Add trasparency and make any values less than the minimum transparent
        rgba_data[..., -1] = 0.4 # Set alpha channel for transparency
        rgba_data[variable_data <= vmin, -1] = 0 # Make values less than vmin fully transparent

        # Save the image
        plt.imsave(image_path, rgba_data, format='png')

    # Function to fetch and plot weather data for a given region
    def fetch_and_plot_weather(year, month, day, lat_range, lon_range, variables=variables, model="ACCESS-CM2", scenario="ssp585"):
        weather_data = fetch_weather_data(year, month, day, lat_range, lon_range, model, scenario)
        heatmap_html = []

        for variable in variables:
            data = weather_data[variable]["value"]
            lat_start, lon_start = lat_range[0], lon_range[0]

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    lat = lat_start + i * LAT_STEP
                    lon = lon_start + j * LON_STEP
                    temperature = float(data[i, j])
                    heatmap_html.append((lat, lon, temperature))

            # Filter out points where temperature is NaN
            filtered_points = [point for point in heatmap_html if not math.isnan(point[2])]

            # Sort the points by temperature in descending order
            sorted_points = sorted(filtered_points, key=lambda x: x[2], reverse=True)

            # Select the top 10 points
            top_10_points = sorted_points[:10]

            # Only create the heatmap if it does not exist already, or we are debugging
            if not os.path.exists(heatmap_image_path) or debug:
                print('hello world')
                plot_weather_data(weather_data, variable, -30, 50, heatmap_image_path)  # TODO: Set the vmin and vmax for different variables

        return heatmap_html, top_10_points

    # Fetch and plot weather data
    heatmap_html, top_10_points = fetch_and_plot_weather(year, month, day, (min_lat, max_lat), (min_lon, max_lon), variables=variables)

    # Convert the list to a numpy array
    data = np.array(heatmap_html)

    # Optionally convert back to list
    heatmap_html = data.tolist()

    # Convert heatmap_html to numpy arrays
    lats, lons, temperatures = np.array(heatmap_html).T

    # Calculate min and max temperatures excluding NaN values
    min_temp = np.nanmin(temperatures)
    max_temp = np.nanmax(temperatures)

    # Round down to nearest multiple of 5
    min_temp = np.floor(min_temp / 5) * 5

    # Round up to nearest multiple of 5
    max_temp = np.ceil(max_temp / 5) * 5

    # Return necessary data
    return lats, lons, min_temp, max_temp, heatmap_image_path, top_10_points, heatmap_html