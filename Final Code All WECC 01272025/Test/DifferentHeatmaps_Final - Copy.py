# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:35:08 2024

@author: Saleh
"""


import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import OpenVisus as ov
import numpy as np
import math
import os
from datetime import date, timedelta
from matplotlib.colors import LinearSegmentedColormap


# Store data that has already been loaded
loaded_datasets: dict[str, any] = {}
loaded_timesteps: dict[tuple[str, int], any] = {}


def plot_HeatMap_diff(year, month, day, min_lat, max_lat, min_lon, max_lon, num_days, heatmap_index):
    # Enable to reload images even if it already exists on disk
    debug = False
    
    # Mapping heatmap_index to heatmap_index_des
    heatmap_descriptions = {
        0: 'MaxTemp',
        1: 'MeanMax',
        2: 'MeanMaxDepNormal',
        3: 'DaysMaxTemp',
        4: 'AvgTemp',
        5: 'MeanAvg',
        6: 'MeanAvgDepNormal',
        7: 'DaysAvgTemp',
        8: 'MinTemp',
        9: 'MeanMin',
        10: 'MeanMinDepNormal',
        11: 'DaysMinTemp'
    }

    # Get the description based on the heatmap_index
    heatmap_index_des = heatmap_descriptions.get(heatmap_index, 'Unknown')
    
    
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
        
        if heatmap_index in [1, 2, 3]:
            variable_names = {
                "tasmax": "Daily Maximum Near-Surface Air Temperature"
            }
            # Define normal values for each variable (replace with actual normal values)
            normal_values = {
                "tasmax": 30.0  # Example normal value for tasmax in °C
            }
        elif heatmap_index in [5, 6, 7]:
            variable_names = {
                "tas": "Daily Near-Surface Air Temperature"
            }
            # Define normal values for each variable (replace with actual normal values)
            normal_values = {
                "tas": 20.0  # Example normal value for tas in °C
            }
            
        elif heatmap_index in [9, 10, 11]:
            variable_names = {
                "tasmin": "Daily Minimum Near-Surface Air Temperature"
            }
            # Define normal values for each variable (replace with actual normal values)
            normal_values = {
                "tasmin": 5.0  # Example normal value for tasmin in °C
            }


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

        return Weather_data, normal_values, variable_names
   
    # Main loop to fetch data for each day in the range
    Weather_data = []
    current_date = date(year, month, day)
    for _ in range(num_days):
        weather_data, normal_values, variable_names = fetch_weather_data(current_date.year, current_date.month, current_date.day, (min_lat, max_lat), (min_lon, max_lon))
        Weather_data.append(weather_data)
        current_date += timedelta(days=1)
    
    if heatmap_index in [1, 5, 9]:
        # Calculate the average values for each variable
        Weather_data_average = {}
        for variable in Weather_data[0]:
            values = np.array([day_data[variable]['value'] for day_data in Weather_data if day_data[variable]['value'] is not None])
            average_value = np.mean(values, axis=0)
            Weather_data_average[variable] = {
                "description": Weather_data[0][variable]['description'],
                "value": average_value
            }
        
        final_data = Weather_data_average
    elif heatmap_index in [2, 6, 10]:
        # Calculate the average values for each variable
        Weather_data_average = {}
        for variable in Weather_data[0]:
            values = np.array([day_data[variable]['value'] for day_data in Weather_data if day_data[variable]['value'] is not None])
            average_value = np.mean(values, axis=0)
            Weather_data_average[variable] = {
                "description": Weather_data[0][variable]['description'],
                "value": average_value
            }
            
        # Calculate the mean departure from normal
        Weather_data_departure = {}
        for variable in Weather_data_average:
            normal_value = normal_values.get(variable, 0)  # Replace 0 with an appropriate default if needed
            departure_values = np.array([day_data[variable]['value'] - normal_value for day_data in Weather_data if day_data[variable]['value'] is not None])
            mean_departure_value = np.mean(departure_values, axis=0)
            Weather_data_departure[variable] = {
                "description": Weather_data[0][variable]['description'] + " (Departure from Normal)",
                "value": mean_departure_value
            }
        
        final_data = Weather_data_departure
        
    elif heatmap_index in [3, 7, 11]:
        
        # Calculate the average values for each variable
        Weather_data_average = {}
        for variable in Weather_data[0]:
            values = np.array([day_data[variable]['value'] for day_data in Weather_data if day_data[variable]['value'] is not None])
            average_value = np.mean(values, axis=0)
            Weather_data_average[variable] = {
                "description": Weather_data[0][variable]['description'],
                "value": average_value
            }
            
            
        # Calculate the number of days with maximum temperature >= 90°F (32.2°C)
        threshold_celsius = 32.2  # 90°F in Celsius
        days_above_90F_matrix = np.zeros_like(Weather_data[0][variable]["value"], dtype=int)
        
        for variable in Weather_data_average:
            for day_data in Weather_data:
                values = day_data[variable]["value"]
                
                if values is not None:
                    # Check for NaN values
                    nan_mask = np.isnan(values)
                    
                    # Where values is not NaN, check if it exceeds the threshold
                    days_above_90F_matrix += np.where(~nan_mask, values >= threshold_celsius, 0)
                    
                    # Set corresponding positions to NaN where values is NaN
                    days_above_90F_matrix = np.where(nan_mask, np.nan, days_above_90F_matrix)
            
            days_above90 = {
                variable: {
                    "description": Weather_data[0][variable]['description'] + " (# Days >= 90°F)",
                    "value": days_above_90F_matrix
                }
            }
            
        final_data = days_above90
        
       
      
    # Function to plot weather data with Matplotlib
    def plot_weather_data(final_data, variable, vmin, vmax, image_path, heatmap_index):
        

        if heatmap_index in [3, 7, 11]:
            
            data = final_data[variable]["value"]
            data = np.flip(data, 0)
            
            norm = Normalize(vmin=vmin, vmax=vmax)
            normalized_data = norm(data)

            # Preserve NaN values after normalization
            normalized_data = np.where(np.isnan(data), np.nan, normalized_data)


            # Apply the colormap on the normalized data
            colormap = plt.cm.YlOrRd
            rgba_data = colormap(normalized_data)

            # Add trasparency and make any values less than the minimum transparent
            rgba_data[..., -1] = 0.4
            rgba_data[np.isnan(data), -1] = 0 # Make NaN values fully transparent

            # Save the image
            plt.imsave(image_path, rgba_data, format='png')
        else:
            
            data = final_data[variable]["value"]
            # If there are any nan values, replace them with a low value
            data = np.nan_to_num(data, nan=-255)
            data = np.flip(data, 0)
            
            # Normalize the data between the provided min and max values
            norm = Normalize(vmin=vmin, vmax=vmax)
            normalized_data = norm(data)
            
            
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
            rgba_data[data <= vmin, -1] = 0 # Make values less than vmin fully transparent

            # Save the image
            plt.imsave(image_path, rgba_data, format='png')

    # Function to fetch and plot weather data for a given region
    def fetch_and_plot_weather(year, month, day, lat_range, lon_range, variables, model="ACCESS-CM2", scenario="ssp585"):
        heatmap_html = []

        for variable in variables:
            data = final_data[variable]["value"]
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

            if heatmap_index in [3, 7 ,11]:
                # Compute dynamic vmin and vmax based on data range
                vmin = np.nanmin(data)
                vmax = np.nanmax(data)
            elif heatmap_index in [2, 6 ,10]:
                # Define fixed temperature range for color mapping
                vmin = -15  # minimum temperature in your standard range
                vmax = 15   # maximum temperature in your standard range
            else:
                # Define fixed temperature range for color mapping
                vmin = -30  # minimum temperature in your standard range
                vmax = 50   # maximum temperature in your standard range
                
            # Only create the heatmap if it does not exist already, or we are debugging
            if not os.path.exists(heatmap_image_path) or debug:
                print('Error')
                plot_weather_data(final_data, variable, vmin, vmax, heatmap_image_path, heatmap_index)  # TODO: Set the vmin and vmax for different variables

        return heatmap_html, top_10_points



    # Fetch and plot weather data
    heatmap_html, top_10_points = fetch_and_plot_weather(year, month, day, (min_lat, max_lat), (min_lon, max_lon), variables=variable_names)

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
    return lats, lons, min_temp, max_temp, heatmap_image_path, top_10_points, heatmap_html, final_data, Weather_data




def generate_heatmap_diff(start_date, num_days, heatmap_index):
    current_date = start_date
    heatmaps = []

    year, month, day = start_date.year, start_date.month, start_date.day

    lats, lons, min_temp, max_temp, heatmap_image_path, top_10_points, heatmap_html, final_data, Weather_data = plot_HeatMap_diff(year, month, day, min_lat, max_lat, min_lon, max_lon, num_days, heatmap_index)

    heatmaps.append({
        'date': current_date.strftime('%Y-%m-%d'),
        'image_path': heatmap_image_path,
        'top_10_points': top_10_points
    })
    
    return heatmaps
 



# Calculate the min and max latitude and longitude values
min_lat = 30
max_lat = 58
min_lon = -130
max_lon = -100

start_date = date(2020, 7, 21)
end_date = date(2020, 7, 24)
num_days = (end_date - start_date).days + 1    


# heatmap_index = "Maximum Temperature # Days >= 90°F"
# heatmap_index = "Mean Maximum Temperature Departure from Normal"
# heatmap_index = "Mean Maximum Temperature"

# heatmap_index = "Average Temperature # Days >= 90°F"
# heatmap_index = "Mean Average Temperature Departure from Normal"
# heatmap_index = "Mean Average Temperature"

heatmap_index = 2
# heatmap_index = "Mean Minimum Temperature Departure from Normal"
# heatmap_index = "Mean Minimum Temperature"




year, month, day = start_date.year, start_date.month, start_date.day



heatmaps = generate_heatmap_diff(start_date, num_days, heatmap_index)


