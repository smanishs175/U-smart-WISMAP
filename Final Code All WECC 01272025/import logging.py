import logging
import matplotlib.pyplot as plt
import numpy as np
import OpenVisus as ov
from scipy.interpolate import griddata
import heapq

def plot_HeatMap(year, month, day, min_lat, max_lat, min_lon, max_lon):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Constants for lat/lon to index conversion
    LAT_START = -59.875
    LON_START = 0.125
    LAT_STEP = 0.25
    LON_STEP = 0.25

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
    def fetch_weather_data(year, month, day, lat_range, lon_range, model="ACCESS-CM2", scenario="ssp585", variables=("tasmax",)):
        # Calculate the day of the year
        day_of_the_year = calculate_day_of_year(year, month, day)

        variable_names = {
            "tasmax": "Daily Maximum Near-Surface Air Temperature"
            # Additional variables can be added here with descriptions
        }

        scenario = "historical" if year < 2015 else scenario
        timestep = year * 365 + day_of_the_year

        Weather_data = {}

        for variable in variables:
            try:
                dataset_name = f"{variable}_day_{model}_{scenario}_r1i1p1f1_gn"
                dataset_url = f"http://atlantis.sci.utah.edu/mod_visus?dataset={dataset_name}&cached=1"

                db = ov.LoadDataset(dataset_url)
                data = db.read(time=timestep, quality=0)

                lat_indices = np.arange(round((lat_range[0] - LAT_START) / LAT_STEP), round((lat_range[1] - LAT_START) / LAT_STEP) + 1)
                lon_indices = np.arange(round((lon_range[0] - LON_START) / LON_STEP), round((lon_range[1] - LON_START) / LON_STEP) + 1)

                data_subset = data[lat_indices[:, None], lon_indices]

                if variable == "tasmax":  # Adjust temperature conversion logic here as needed
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
    def plot_weather_data(weather_data, lat_range, lon_range, variable):
        data = weather_data[variable]["value"]
        variable_name = weather_data[variable]["description"]

        fig, ax = plt.subplots(figsize=(9, 3))
        im = ax.imshow(np.flip(data, 0), cmap='Reds', vmin=28, vmax=42)  # Adjust vmin and vmax as per your data range

        xticks = np.linspace(0, data.shape[1] - 1, num=5).astype(int)
        yticks = np.linspace(0, data.shape[0] - 1, num=5).astype(int)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([str(x) for x in np.linspace(lon_range[0], lon_range[1], num=5)])
        ax.set_yticklabels([str(x) for x in np.linspace(lat_range[0], lat_range[1], num=5)])
        ax.set_xlabel('Longitude [degrees east]')
        ax.set_ylabel('Latitude [degrees north]')
        ax.set_title(f"{variable_name} in Celsius")

        cbar = fig.colorbar(im, ax=ax, label='Temperature [°C]')
        cbar.set_label('Temperature [°C]')
        plt.show()

        # Return the data for comparison
        return np.flip(data, 0)

    # Function to fetch and plot weather data for a given region
    def fetch_and_plot_weather(year, month, day, lat_range, lon_range, variables=("tasmax",), model="ACCESS-CM2", scenario="ssp585"):
        weather_data = fetch_weather_data(year, month, day, lat_range, lon_range, model, scenario, variables)
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

            heatmap_matplot = plot_weather_data(weather_data, lat_range, lon_range, variable)

        return heatmap_matplot, heatmap_html


    # Fetch and plot weather data
    heatmap_matplot, heatmap_html = fetch_and_plot_weather(year, month, day, (min_lat, max_lat), (min_lon, max_lon))
    
    # Convert the list to a numpy array
    data = np.array(heatmap_html)

    def fill_nan(data):
        n = len(data)
        for i in range(n):
            # Check if current value is NaN
            if np.isnan(data[i, 2]):
                # Search for the nearest valid value
                prev_valid = next((j for j in range(i-1, -1, -1) if not np.isnan(data[j, 2])), None)
                next_valid = next((j for j in range(i+1, n) if not np.isnan(data[j, 2])), None)
                
                if prev_valid is not None and next_valid is not None:
                    data[i, 2] = (data[prev_valid, 2] + data[next_valid, 2]) / 2
                elif prev_valid is not None:
                    data[i, 2] = data[prev_valid, 2]
                elif next_valid is not None:
                    data[i, 2] = data[next_valid, 2]
        return data

    # Apply the fill_nan function
    filled_data = fill_nan(data)

    # Optionally convert back to list
    heatmap_html = filled_data.tolist()
    
    # Convert heatmap_html to numpy arrays
    lats, lons, temperatures = np.array(heatmap_html).T

    # Calculate min and max temperatures
    min_temp = np.floor(min(temperatures) / 5) * 5  # Round down to nearest multiple of 5
    max_temp = np.ceil(max(temperatures) / 5) * 5  # Round up to nearest multiple of 5

    # Extract the 10 highest temperature points
    top_10_temps_indices = heapq.nlargest(10, range(len(temperatures)), key=temperatures.__getitem__)
    top_10_points = [(lats[i], lons[i], temperatures[i]) for i in top_10_temps_indices]

    # Define the grid for interpolation
    grid_lon, grid_lat = np.mgrid[min(lons):max(lons):200j, min(lats):max(lats):200j]

    # Interpolate the temperature values onto the grid
    grid_z = griddata((lons, lats), temperatures, (grid_lon, grid_lat), method='cubic')

    # Create a plot without showing it
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap = ax.imshow(grid_z, extent=(min(lons), max(lons), min(lats), max(lats)), origin='lower', cmap='YlOrRd', alpha=0.6)
    plt.axis('off')  # Hide the axes

    # Plot the top 10 points
    for point in top_10_points:
        lat, lon, temp = point
        ax.plot(lon, lat, 'bo')  # Blue dot
        ax.text(lon, lat, f'{temp:.1f}°C', color='blue', fontsize=8, ha='left')

    # Save the figure as a transparent image
    heatmap_image_path = "static/heatmap_image.png"
    plt.savefig(heatmap_image_path, bbox_inches='tight', pad_inches=0, transparent=True)

    # Close the plot to free memory
    plt.close(fig)

    return lats, lons, min_temp, max_temp, heatmap_image_path, top_10_points, heatmap_html



# Calculate the min and max latitude and longitude values
min_lat = 30
max_lat = 58
min_lon = -130
max_lon = -90
year = 2020
month = 7
day = 21


lats, lons, min_temp, max_temp, heatmap_image_path, top_10_points, heatmap_html = plot_HeatMap(year, month, day, min_lat, max_lat, min_lon, max_lon)
