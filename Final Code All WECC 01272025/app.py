from flask import Flask, render_template, request, redirect, url_for, jsonify
import geopandas as gpd
import folium
from shapely.wkt import loads
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
from branca.element import CssLink
from weather_data_extractor import get_weather_data
from weather_data_extractor_BA import get_weather_data_BA
from folium.plugins import MarkerCluster
from folium.raster_layers import ImageOverlay
from HeatMapPlot_FinalAll import plot_HeatMap
from DifferentHeatmaps_Final import plot_HeatMap_diff
import logging
import os
from datetime import datetime
from datetime import timedelta
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from shapely.geometry import shape, Point, LineString
import math
from shapely.geometry import Polygon
import random
import json



app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG


year = 2020
month = 7
day = 21

# Function to safely load WKT strings
def safe_loads(wkt_str):
    try:
        return loads(wkt_str)
    except Exception as e:
        print(f"Error loading geometry: {wkt_str} -> {e}")
        return None

# Function to read and process the first 100 rows of the Excel files
def read_and_process_excel(file_path):
    # Read only the first 100 rows
    df = pd.read_excel(file_path)
    if 'geometry' not in df.columns:
        raise ValueError(f"Missing 'geometry' column in the data: {file_path}")
    
    df['geometry'] = df['geometry'].apply(safe_loads)
    df = df.dropna(subset=['geometry'])  # Drop rows with invalid geometries
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.crs = 'epsg:4326'

    # Filter out rows where coordinates are zero for Points and LineStrings
    def is_valid_geometry(geometry):
        if isinstance(geometry, Point):
            return not (geometry.x == 0 or geometry.y == 0)
        elif isinstance(geometry, LineString):
            # Check if any coordinate in the LineString has zero values
            return all(x != 0 and y != 0 for x, y in geometry.coords)
        return True

    gdf = gdf[gdf.geometry.apply(is_valid_geometry)]
    return gdf


# Read Transmission lines data
transmission_line_gdf = read_and_process_excel("Data/WECC data/merged_branch_data.xlsx")
required_columns = ['geometry', 'RATE1', 'NAME', 'Line Status', 'RATE2']
missing_columns = [col for col in required_columns if col not in transmission_line_gdf.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in the data: {missing_columns}")

# Read additional data
gen_gdf = read_and_process_excel("Data/WECC data/merged_gen_data.xlsx")
bus_gdf = read_and_process_excel("Data/WECC data/merged_bus_data.xlsx")
substation_gdf = read_and_process_excel("Data/WECC data/merged_substation_data.xlsx")
load_gdf = read_and_process_excel("Data/WECC data/merged_load_data.xlsx")

# Define voltage level mapping based on the numeric 'RATE1' values
def map_voltage(rate1_value):
    if pd.isna(rate1_value):
        return pd.NA  # Handle NaN values as pd.NA
    # Ensure that rate1_value is an integer
    return int(rate1_value)

# Apply the mapping function to the DataFrame
transmission_line_gdf['Voltage'] = transmission_line_gdf['RATE1'].apply(map_voltage).astype(pd.Int64Dtype())

# Calculate the min and max latitude and longitude values
min_lat = 30
max_lat = 58
min_lon = -130
max_lon = -100

# Define latitude and longitude ranges with a step size of 1
lat_range = np.arange(min_lat - 4, max_lat + 4, 1)
lon_range = np.arange(min_lon - 3, max_lon + 3, 1)

# Create a list of coordinates within the specified range
coordinates = [(lat, lon) for lat in lat_range for lon in lon_range]

# Function to fill NaN values
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

# Function to fetch weather data from NSDF
def fetch_weather_data_for_point(year, month, day, latitude, longitude):
    try:
        weather_data = get_weather_data(year=year, month=month, day=day, latitude=latitude, longitude=longitude)
        return (
            weather_data["tasmax"]["value"],  # Max temperature
            weather_data["tas"]["value"],     # Avg temperature
            weather_data["tasmin"]["value"],  # Min temperature
            weather_data["hurs"]["value"],    # Relative humidity
            weather_data["huss"]["value"],    # Specific humidity
            weather_data["rlds"]["value"],    # Longwave radiation
            weather_data["rsds"]["value"],    # Shortwave radiation
            weather_data["pr"]["value"],      # Precipitation
            weather_data["sfcWind"]["value"]  # Wind speed
        )
    except Exception as e:
        print(f"Error fetching weather data for point ({latitude}, {longitude}): {e}")
        return ["No data available"] * 9


def calculate_min_max(data):
    # Define the parameter indices in the tuple
    param_indices = {
        'Max_temp': 0,
        'Avg_temp': 1,
        'Min_temp': 2,
        'Rel_hum': 3,
        'Spec_hum': 4,
        'Long_rad': 5,
        'Short_rad': 6,
        'Perc': 7,
        'Wind_speed': 8
    }

    min_values = {}
    max_values = {}

    for param, index in param_indices.items():
        values = []
        dates = []

        for entry in data:
            values.append(entry['weather_data'][index])
            dates.append(entry['date'])

        min_val = min(values)
        max_val = max(values)
        min_date = dates[values.index(min_val)]
        max_date = dates[values.index(max_val)]

        min_values[param] = (min_val, min_date)
        max_values[param] = (max_val, max_date)

    return min_values, max_values


def fetch_weather_data_for_range(year, month, day, lat, lon, num_days, type, send_info):
    if type not in ['load', 'gen', 'substation', 'node', 'line']:
        raise ValueError("Unsupported type")
    if type == 'line':
        CL = send_info  # Unpack the send_info tuple
        data = []
        start_date = datetime(year, month, day)
        
        # Initialize values for day one
        day_one_data = {
            'date': start_date,
            'weather_data': fetch_weather_data_for_point(start_date.year, start_date.month, start_date.day, lat, lon)
        }
        day_one_max_temp = day_one_data['weather_data'][0]
        
        data.append({
            'date': start_date,
            'weather_data': day_one_data['weather_data'],
            'CL_day': CL,
        })
        
        alpha_l = 1
        T_RL = 35
        
        for i in range(1, num_days):
            current_date = start_date + timedelta(days=i)
            weather_data = fetch_weather_data_for_point(current_date.year, current_date.month, current_date.day, lat, lon)
            Max_temp_current = weather_data[0]
             
            # Calculate line capacity for the current day
            CL_day_current = CL * alpha_l * math.sqrt(T_RL/Max_temp_current)

            # Check the conditions and adjust CL_day_current if necessary
            if CL_day_current > CL or CL_day_current < 0:
                CL_day_current = CL
            
            data.append({
                'date': current_date,
                'weather_data': weather_data,
                'CL_day': CL_day_current,
            })
            
            
    if type == 'load':
        PL, QL = send_info  # Unpack the send_info tuple
        data = []
        start_date = datetime(year, month, day)
        
        # Initialize values for day one
        day_one_data = {
            'date': start_date,
            'weather_data': fetch_weather_data_for_point(start_date.year, start_date.month, start_date.day, lat, lon)
        }
        day_one_max_temp = day_one_data['weather_data'][0]
        
        data.append({
            'date': start_date,
            'weather_data': day_one_data['weather_data'],
            'PL_day': PL,
            'QL_day': QL
        })
        
        for i in range(1, num_days):
            current_date = start_date + timedelta(days=i)
            weather_data = fetch_weather_data_for_point(current_date.year, current_date.month, current_date.day, lat, lon)
            Max_temp_current = weather_data[0]
            
            # Calculate PL_day and QL_day for the current day
            PL_day_current = PL * (1 + 0.01*(5.33 - 0.067 * lon) * (Max_temp_current - day_one_max_temp))
            # Ensure PL_day_current is not negative
            PL_day_current = max(0, PL_day_current)

            QL_day_current = QL * (1 + 0.01*(5.33 - 0.067 * lon) * (Max_temp_current - day_one_max_temp))
            
            data.append({
                'date': current_date,
                'weather_data': weather_data,
                'PL_day': PL_day_current,
                'QL_day': QL_day_current
            })
    elif type == 'gen':
        Pgen, Qgen, gen_type = send_info  # Unpack the send_info tuple
        data = []
        start_date = datetime(year, month, day)
        eff = 1
        
        # Initialize values for day one
        day_one_data = {
            'date': start_date,
            'weather_data': fetch_weather_data_for_point(start_date.year, start_date.month, start_date.day, lat, lon)
        }
        day_one_max_temp = day_one_data['weather_data'][0]
        
        data.append({
            'date': start_date,
            'weather_data': day_one_data['weather_data'],
            'Pgen_day': Pgen,
            'Qgen_day': Qgen,
            'Effgen_day': eff
        })
        
        for i in range(1, num_days):
            current_date = start_date + timedelta(days=i)
            weather_data = fetch_weather_data_for_point(current_date.year, current_date.month, current_date.day, lat, lon)
            Max_temp_current = weather_data[0]
            Wind_speed_current = weather_data[8]
            Long_rad_current = weather_data[5]
            V_cutin = 3
            V_cutout = 25
            V_rated = 12
            T_th_PV = 35
            ro_sf = 0.02
            T_th_gen = 40
            ro_th = 0.031
            eff_no = 0.6
            
            # Calculate Pgen_day and Qgen_day for the current day
            if gen_type == 'WT-Onshore':
                if Wind_speed_current < V_cutin or Wind_speed_current > V_cutout:
                    Pgen_day_current = 0
                elif V_cutin <= Wind_speed_current < V_rated:
                    Pgen_day_current = Pgen * ((Wind_speed_current - V_cutin) / (V_rated - V_cutin))
                else:
                    Pgen_day_current = Pgen
                
                eff = 1
                Qgen_day_current = Qgen
        
            elif gen_type in ['SolarPV-Tracking', 'SolarPV-NonTracking']:
                if Max_temp_current <= T_th_PV:
                    eff = 1
                else:
                    eff = eff_no * (1 - ro_sf * (Max_temp_current - T_th_PV))
                
                Pgen_day_current = Pgen * eff
                Qgen_day_current = Qgen
        
            else:
                if Max_temp_current <= T_th_gen:
                    eff = 1
                else:
                    eff = 1 - ro_th * (Max_temp_current - T_th_gen)
                
                Pgen_day_current = Pgen * eff
                Qgen_day_current = Qgen * eff
                
            data.append({
                'date': current_date,
                'weather_data': weather_data,
                'Pgen_day': Pgen_day_current,
                'Qgen_day': Qgen_day_current,
                'Effgen_day': eff
            })
        
    elif type in ['substation', 'node']:
        data = []
        start_date = datetime(year, month, day)
        
        # Initialize values for day one
        day_one_data = {
            'date': start_date,
            'weather_data': fetch_weather_data_for_point(start_date.year, start_date.month, start_date.day, lat, lon)
        }
        day_one_max_temp = day_one_data['weather_data'][0]
        
        data.append({
            'date': start_date,
            'weather_data': day_one_data['weather_data'],
        })
        
        for i in range(1, num_days):
            current_date = start_date + timedelta(days=i)
            weather_data = fetch_weather_data_for_point(current_date.year, current_date.month, current_date.day, lat, lon)
            Max_temp_current = weather_data[0]
            
            data.append({
                'date': current_date,
                'weather_data': weather_data,
            })
   
        
    return data


def calculate_average_data(data_st, data_end):
    if len(data_st) != len(data_end):
        raise ValueError("Data length mismatch between start and end points")
    
    data_avg = []
    
    for day_st, day_end in zip(data_st, data_end):
        if day_st['date'] != day_end['date']:
            raise ValueError("Date mismatch between start and end points")
        
        avg_weather_data = [
            (st + end) / 2 for st, end in zip(day_st['weather_data'], day_end['weather_data'])
        ]
        avg_CL_day = (day_st['CL_day'] + day_end['CL_day']) / 2
        
        data_avg.append({
            'date': day_st['date'],
            'weather_data': avg_weather_data,
            'CL_day': avg_CL_day,
        })
    
    return data_avg


def create_time_series_plot(data, type):
    dates = [entry['date'].strftime('%Y-%m-%d') for entry in data]
    Max_temp = [entry['weather_data'][0] for entry in data]
    Avg_temp = [entry['weather_data'][1] for entry in data]
    Min_temp = [entry['weather_data'][2] for entry in data]
    Rel_hum = [entry['weather_data'][3] for entry in data]
    Spec_hum = [entry['weather_data'][4] for entry in data]
    Long_rad = [entry['weather_data'][5] for entry in data]
    Short_rad = [entry['weather_data'][6] for entry in data]
    Perc = [entry['weather_data'][7] for entry in data]
    Wind_speed = [entry['weather_data'][8] for entry in data]

    # Define specs based on plot type
    specs = [
        [{'type': 'xy'}, {'type': 'xy', 'secondary_y': True}],
        [{'type': 'xy'}, {'type': 'xy'}],
        [{'type': 'xy'}, {'type': 'xy', 'secondary_y': True} if type in ['load', 'gen'] else None]
    ]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Max, Avg, Min Temperature',
            'Relative & Specific Humidity',
            'Longwave & Shortwave Radiation',
            'Precipitation',
            'Wind Speed',
            'PL_day & QL_day' if type == 'load' else 'Pgen_day & Qgen_day & Effgen_day' if type == 'gen' else ''
        ),
        shared_xaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.07,
        specs=specs
    )

    # Add traces
    fig.add_trace(go.Scatter(x=dates, y=Max_temp, mode='lines+markers', name='Max Temperature'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Avg_temp, mode='lines+markers', name='Avg Temperature'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Min_temp, mode='lines+markers', name='Min Temperature'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Rel_hum, mode='lines+markers', name='Relative Humidity'), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=Spec_hum, mode='lines+markers', name='Specific Humidity'), row=1, col=2, secondary_y=True)
    fig.add_trace(go.Scatter(x=dates, y=Long_rad, mode='lines+markers', name='Longwave Radiation'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Short_rad, mode='lines+markers', name='Shortwave Radiation'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Perc, mode='lines+markers', name='Precipitation'), row=2, col=2)
    fig.add_trace(go.Scatter(x=dates, y=Wind_speed, mode='lines+markers', name='Wind Speed'), row=3, col=1)

    if type == 'load':
        PL_day = [entry.get('PL_day', 0) for entry in data]
        QL_day = [entry.get('QL_day', 0) for entry in data]
        fig.add_trace(go.Scatter(x=dates, y=PL_day, mode='lines+markers', name='PL_day'), row=3, col=2)
        fig.add_trace(go.Scatter(x=dates, y=QL_day, mode='lines+markers', name='QL_day'), row=3, col=2)
    
    if type == 'gen':
        Pgen_day = [entry.get('Pgen_day', 0) for entry in data]
        Qgen_day = [entry.get('Qgen_day', 0) for entry in data]
        Effgen_day = [entry.get('Effgen_day', 0) for entry in data]
        fig.add_trace(go.Scatter(x=dates, y=Pgen_day, mode='lines+markers', name='Active Power'), row=3, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=dates, y=Qgen_day, mode='lines+markers', name='Reactive Power'), row=3, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=dates, y=Effgen_day, mode='lines+markers', name='Efficiency'), row=3, col=2, secondary_y=True)

    fig.update_layout(
        xaxis=dict(tickformat='%Y-%m-%d', dtick="D1"),
        height=1000,
        width=1500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            font=dict(
                family="Times New Roman",
                size=12,
                color="black"
            ),
            traceorder='reversed',
            itemsizing='constant'
        ),
        font=dict(
            family="Times New Roman",
            size=12,
            color="black"
        )
    )

    fig.update_yaxes(title_text='Temperature (°C)', row=1, col=1)
    fig.update_yaxes(title_text='Relative Humidity (%)', row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text='Specific Humidity (g/kg)', row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text='Radiation (W/m²)', row=2, col=1)
    fig.update_yaxes(title_text='Precipitation (mm)', row=2, col=2)
    fig.update_yaxes(title_text='Wind Speed (m/s)', row=3, col=1)
    
    if type == 'load':
        fig.update_yaxes(title_text='Demand (MW/MVar)', row=3, col=2)
        
    if type == 'gen':
        fig.update_yaxes(title_text='Power (MW/MVar)', row=3, col=2, secondary_y=False)
        fig.update_yaxes(title_text='Efficiency (%)', row=3, col=2, secondary_y=True)

    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_xaxes(title_text='Date', row=3, col=2 if type in ['load', 'gen'] else 1)
    fig.update_xaxes(title_font=dict(size=14, family="Times New Roman"), tickfont=dict(size=12, family="Times New Roman"))
    fig.update_yaxes(title_font=dict(size=14, family="Times New Roman"), tickfont=dict(size=12, family="Times New Roman"))

    plot_html = plot(fig, output_type='div')
    min_values, max_values = calculate_min_max(data)
    
    return plot_html, min_values, max_values

def create_time_series_plot_line(data_st, data_end, data_avg):
    dates = [entry['date'].strftime('%Y-%m-%d') for entry in data_st]
    Max_temp_st = [entry['weather_data'][0] for entry in data_st]
    Avg_temp_st = [entry['weather_data'][1] for entry in data_st]
    Min_temp_st = [entry['weather_data'][2] for entry in data_st]
    Rel_hum_st = [entry['weather_data'][3] for entry in data_st]
    Spec_hum_st = [entry['weather_data'][4] for entry in data_st]
    Long_rad_st = [entry['weather_data'][5] for entry in data_st]
    Short_rad_st = [entry['weather_data'][6] for entry in data_st]
    Perc_st = [entry['weather_data'][7] for entry in data_st]
    Wind_speed_st = [entry['weather_data'][8] for entry in data_st]
    CL_day_st = [entry.get('CL_day', 0) for entry in data_st]

    
    Max_temp_end = [entry['weather_data'][0] for entry in data_end]
    Avg_temp_end = [entry['weather_data'][1] for entry in data_end]
    Min_temp_end = [entry['weather_data'][2] for entry in data_end]
    Rel_hum_end = [entry['weather_data'][3] for entry in data_end]
    Spec_hum_end = [entry['weather_data'][4] for entry in data_end]
    Long_rad_end = [entry['weather_data'][5] for entry in data_end]
    Short_rad_end = [entry['weather_data'][6] for entry in data_end]
    Perc_end = [entry['weather_data'][7] for entry in data_end]
    Wind_speed_end = [entry['weather_data'][8] for entry in data_end]
    CL_day_end = [entry.get('CL_day', 0) for entry in data_end]
    
    Max_temp_avg = [entry['weather_data'][0] for entry in data_avg]
    Avg_temp_avg = [entry['weather_data'][1] for entry in data_avg]
    Min_temp_avg = [entry['weather_data'][2] for entry in data_avg]
    Rel_hum_avg = [entry['weather_data'][3] for entry in data_avg]
    Spec_hum_avg = [entry['weather_data'][4] for entry in data_avg]
    Long_rad_avg = [entry['weather_data'][5] for entry in data_avg]
    Short_rad_avg = [entry['weather_data'][6] for entry in data_avg]
    Perc_avg = [entry['weather_data'][7] for entry in data_avg]
    Wind_speed_avg = [entry['weather_data'][8] for entry in data_avg]
    CL_day_avg = [entry.get('CL_day', 0) for entry in data_avg]

    # Define specs based on plot type
    specs = [
        [{'type': 'xy'}, {'type': 'xy', 'secondary_y': True}],
        [{'type': 'xy'}, {'type': 'xy'}],
        [{'type': 'xy'}, {'type': 'xy'}]
    ]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Max, Avg, Min Temperature',
            'Relative & Specific Humidity',
            'Longwave & Shortwave Radiation',
            'Precipitation',
            'Wind Speed',
            'Line Capacity' 
        ),
        shared_xaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.07,
        specs=specs
    )

    # Add traces
    fig.add_trace(go.Scatter(x=dates, y=Max_temp_st, mode='lines+markers', name='Max Temperature ST'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Avg_temp_st, mode='lines+markers', name='Avg Temperature ST'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Min_temp_st, mode='lines+markers', name='Min Temperature ST'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Max_temp_end, mode='lines+markers', name='Max Temperature End'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Avg_temp_end, mode='lines+markers', name='Avg Temperature End'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Min_temp_end, mode='lines+markers', name='Min Temperature End'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Max_temp_avg, mode='lines+markers', name='Max Temperature Avg'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Avg_temp_avg, mode='lines+markers', name='Avg Temperature Avg'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Min_temp_avg, mode='lines+markers', name='Min Temperature Avg'), row=1, col=1)
    
    
    fig.add_trace(go.Scatter(x=dates, y=Rel_hum_st, mode='lines+markers', name='Relative Humidity ST'), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=Spec_hum_st, mode='lines+markers', name='Specific Humidity ST'), row=1, col=2, secondary_y=True)
    fig.add_trace(go.Scatter(x=dates, y=Rel_hum_end, mode='lines+markers', name='Relative Humidity End'), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=Spec_hum_end, mode='lines+markers', name='Specific Humidity End'), row=1, col=2, secondary_y=True)
    fig.add_trace(go.Scatter(x=dates, y=Rel_hum_avg, mode='lines+markers', name='Relative Humidity Avg'), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=Spec_hum_avg, mode='lines+markers', name='Specific Humidity Avg'), row=1, col=2, secondary_y=True)
    
    
    fig.add_trace(go.Scatter(x=dates, y=Long_rad_st, mode='lines+markers', name='Longwave Radiation ST'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Short_rad_st, mode='lines+markers', name='Shortwave Radiation ST'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Long_rad_end, mode='lines+markers', name='Longwave Radiation End'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Short_rad_end, mode='lines+markers', name='Shortwave Radiation End'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Long_rad_avg, mode='lines+markers', name='Longwave Radiation Avg'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Short_rad_avg, mode='lines+markers', name='Shortwave Radiation Avg'), row=2, col=1)
    
    
    fig.add_trace(go.Scatter(x=dates, y=Perc_st, mode='lines+markers', name='Precipitation ST'), row=2, col=2)
    fig.add_trace(go.Scatter(x=dates, y=Perc_end, mode='lines+markers', name='Precipitation End'), row=2, col=2)
    fig.add_trace(go.Scatter(x=dates, y=Perc_avg, mode='lines+markers', name='Precipitation Avg'), row=2, col=2)
    
    
    fig.add_trace(go.Scatter(x=dates, y=Wind_speed_st, mode='lines+markers', name='Wind Speed ST'), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Wind_speed_end, mode='lines+markers', name='Wind Speed End'), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=Wind_speed_avg, mode='lines+markers', name='Wind Speed Avg'), row=3, col=1)
    
    fig.add_trace(go.Scatter(x=dates, y=CL_day_st, mode='lines+markers', name='CL ST'), row=3, col=2)
    fig.add_trace(go.Scatter(x=dates, y=CL_day_end, mode='lines+markers', name='CL End'), row=3, col=2)
    fig.add_trace(go.Scatter(x=dates, y=CL_day_avg, mode='lines+markers', name='CL Avg'), row=3, col=2)


    
    fig.update_layout(
        xaxis=dict(tickformat='%Y-%m-%d', dtick="D1"),
        height=1000,
        width=1500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            font=dict(
                family="Times New Roman",
                size=12,
                color="black"
            ),
            traceorder='reversed',
            itemsizing='constant'
        ),
        font=dict(
            family="Times New Roman",
            size=12,
            color="black"
        )
    )

    fig.update_yaxes(title_text='Temperature (°C)', row=1, col=1)
    fig.update_yaxes(title_text='Relative Humidity (%)', row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text='Specific Humidity (g/kg)', row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text='Radiation (W/m²)', row=2, col=1)
    fig.update_yaxes(title_text='Precipitation (mm)', row=2, col=2)
    fig.update_yaxes(title_text='Wind Speed (m/s)', row=3, col=1)
    fig.update_yaxes(title_text='Capacity (MW)', row=3, col=2)
    

    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_xaxes(title_text='Date', row=3, col=2)
    fig.update_xaxes(title_font=dict(size=14, family="Times New Roman"), tickfont=dict(size=12, family="Times New Roman"))
    fig.update_yaxes(title_font=dict(size=14, family="Times New Roman"), tickfont=dict(size=12, family="Times New Roman"))

    plot_html = plot(fig, output_type='div')
    min_values, max_values = calculate_min_max(data_avg)
    
    return plot_html, min_values, max_values


# Function to format the tooltip content for generators
def gen_popup_content(name, gen_type, pgen, qgen, lat, lon):
    weather_data = fetch_weather_data_for_point(year, month, day, lat, lon)
    if weather_data != ("No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available"):
        max_temp = f"{weather_data[0]:.2f}°C"
        avg_temp = f"{weather_data[1]:.2f}°C"
        min_temp = f"{weather_data[2]:.2f}°C"
        relative_humidity = f"{weather_data[3]:.2f}%"
        specific_humidity = f"{weather_data[4]:.2f}%"
        longwave_radiation = f"{weather_data[5]:.2f} W/m^2"
        shortwave_radiation = f"{weather_data[6]:.2f} W/m^2"
        precipitation = f"{weather_data[7]:.2f} mm"
        wind_speed = f"{weather_data[8]:.2f} m/s"
    else:
        max_temp = avg_temp = min_temp = relative_humidity = specific_humidity = longwave_radiation = shortwave_radiation = precipitation = wind_speed = "No data available"

    return {
        "name": name,
        "gen_type": gen_type,
        "pgen": pgen,
        "qgen": qgen,
        "max_temp": max_temp,
        "avg_temp": avg_temp,
        "min_temp": min_temp,
        "relative_humidity": relative_humidity,
        "specific_humidity": specific_humidity,
        "longwave_radiation": longwave_radiation,
        "shortwave_radiation": shortwave_radiation,
        "precipitation": precipitation,
        "wind_speed": wind_speed
    }


# Function to format the tooltip content for loads
def load_popup_content(name, load_type, pl, ql, lat, lon):
    weather_data = fetch_weather_data_for_point(year, month, day, lat, lon)
    
    def format_value(value, suffix=""):
        return f"{value:.2f}{suffix}" if value is not None else "No data available"
    
    max_temp = format_value(weather_data[0], "°C")
    avg_temp = format_value(weather_data[1], "°C")
    min_temp = format_value(weather_data[2], "°C")
    relative_humidity = format_value(weather_data[3], "%")
    specific_humidity = format_value(weather_data[4], "%")
    longwave_radiation = format_value(weather_data[5], " W/m^2")
    shortwave_radiation = format_value(weather_data[6], " W/m^2")
    precipitation = format_value(weather_data[7], " mm")
    wind_speed = format_value(weather_data[8], " m/s")
    
    return {
        "name": name,
        "load_type": load_type,
        "pl": pl,
        "ql": ql,
        "max_temp": max_temp,
        "avg_temp": avg_temp,
        "min_temp": min_temp,
        "relative_humidity": relative_humidity,
        "specific_humidity": specific_humidity,
        "longwave_radiation": longwave_radiation,
        "shortwave_radiation": shortwave_radiation,
        "precipitation": precipitation,
        "wind_speed": wind_speed
    }

# Function to format the tooltip content for substations
def substation_popup_content(name, substation_type, lat, lon):
    weather_data = fetch_weather_data_for_point(year, month, day, lat, lon)
    if weather_data != ("No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available"):
        max_temp = f"{weather_data[0]:.2f}°C"
        avg_temp = f"{weather_data[1]:.2f}°C"
        min_temp = f"{weather_data[2]:.2f}°C"
        relative_humidity = f"{weather_data[3]:.2f}%"
        specific_humidity = f"{weather_data[4]:.2f}%"
        longwave_radiation = f"{weather_data[5]:.2f} W/m^2"
        shortwave_radiation = f"{weather_data[6]:.2f} W/m^2"
        precipitation = f"{weather_data[7]:.2f} mm"
        wind_speed = f"{weather_data[8]:.2f} m/s"
    else:
        max_temp = avg_temp = min_temp = relative_humidity = specific_humidity = longwave_radiation = shortwave_radiation = precipitation = wind_speed = "No data available"

    return {
        "name": name,
        "substation_type": substation_type,
        "max_temp": max_temp,
        "avg_temp": avg_temp,
        "min_temp": min_temp,
        "relative_humidity": relative_humidity,
        "specific_humidity": specific_humidity,
        "longwave_radiation": longwave_radiation,
        "shortwave_radiation": shortwave_radiation,
        "precipitation": precipitation,
        "wind_speed": wind_speed
    }
           
           
           
# Function to format the tooltip content for buses
def bus_popup_content(name, bus_type, voltage, lat, lon):
    weather_data = fetch_weather_data_for_point(year, month, day, lat, lon)
    if weather_data != ("No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available"):
        max_temp = f"{weather_data[0]:.2f}°C"
        avg_temp = f"{weather_data[1]:.2f}°C"
        min_temp = f"{weather_data[2]:.2f}°C"
        relative_humidity = f"{weather_data[3]:.2f}%"
        specific_humidity = f"{weather_data[4]:.2f}%"
        longwave_radiation = f"{weather_data[5]:.2f} W/m^2"
        shortwave_radiation = f"{weather_data[6]:.2f} W/m^2"
        precipitation = f"{weather_data[7]:.2f} mm"
        wind_speed = f"{weather_data[8]:.2f} m/s"
    else:
        max_temp = avg_temp = min_temp = relative_humidity = specific_humidity = longwave_radiation = shortwave_radiation = precipitation = wind_speed = "No data available"

    return {
        "name": name,
        "bus_type": bus_type,
        "voltage": voltage,
        "max_temp": max_temp,
        "avg_temp": avg_temp,
        "min_temp": min_temp,
        "relative_humidity": relative_humidity,
        "specific_humidity": specific_humidity,
        "longwave_radiation": longwave_radiation,
        "shortwave_radiation": shortwave_radiation,
        "precipitation": precipitation,
        "wind_speed": wind_speed
    }



# Function to add points from GeoDataFrames to the map with the appropriate popup
def get_points_data(gdf, popup_template, data_type):
    points_data = []
    if 'NAME' not in gdf.columns:
        print(f"Missing 'NAME' column in the data.")
        return points_data

    for _, row in gdf.iterrows():
        name = row['NAME']
        if data_type == 'generator':
            name = row['Bus Name']
            gen_type = row['SubType']
            pgen = row.get('PG', 'N/A')
            qgen = row.get('QG', 'N/A')
            point = row['geometry'].coords[0]
            lat = row['geometry'].y
            lon = row['geometry'].x
            popup_content = popup_template(name, gen_type, pgen, qgen, lat, lon)
            points_data.append({
                "location": [point[1], point[0]],
                "popup": popup_content
            })
        elif data_type == 'load':
            load_type = "Load"
            pl = row.get('PL', 'N/A')
            ql = row.get('QL', 'N/A')
            point = row['geometry'].coords[0]
            lat = row['geometry'].y
            lon = row['geometry'].x
            popup_content = popup_template(name, load_type, pl, ql, lat, lon)
            points_data.append({
                "location": [point[1], point[0]],
                "popup": popup_content
            })
        elif data_type == 'substation':
            substation_type = "Substation"
            point = row['geometry'].coords[0]
            lat = row['geometry'].y
            lon = row['geometry'].x
            popup_content = popup_template(name, substation_type, lat, lon)
            points_data.append({
                "location": [point[1], point[0]],
                "popup": popup_content
            })
        elif data_type == 'bus':
            bus_type = "Bus"
            voltage = row.get('Base KV', 'N/A')
            point = row['geometry'].coords[0]
            lat = row['geometry'].y
            lon = row['geometry'].x
            popup_content = popup_template(name, bus_type, voltage, lat, lon)
            points_data.append({
                "location": [point[1], point[0]],
                "popup": popup_content
            })
        else:
            continue
    return points_data


# Add points from other datasets
gen_points_data = get_points_data(gen_gdf, gen_popup_content, 'generator')
bus_points_data = get_points_data(bus_gdf, bus_popup_content, 'bus')
substation_points_data = get_points_data(substation_gdf, substation_popup_content, 'substation')
load_points_data = get_points_data(load_gdf, load_popup_content, 'load')


# Function to generate heatmap and save to static directory
def generate_heatmap(start_date, end_date, heatmap_index):
    current_date = start_date
    heatmaps = []

    while current_date <= end_date:
        year, month, day = current_date.year, current_date.month, current_date.day
        lats, lons, min_temp, max_temp, heatmap_image_path, top_10_points, heatmap_html = plot_HeatMap(year, month, day, min_lat, max_lat, min_lon, max_lon, heatmap_index)
   
        heatmaps.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'image_path': heatmap_image_path,
            'top_10_points': top_10_points
        })

        current_date += timedelta(days=1)

    return heatmaps

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
 
# Function to fetch weather data for each point in the line
def fetch_weather_data_for_line(coords):
    for lat, lon in coords:
        weather_data = fetch_weather_data_for_point(year, month, day, lat, lon)
        if weather_data != ("No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available"):
            return weather_data  # Return as soon as valid data is found
    return ("No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available", "No data available")



def get_line_data(gdf):
    line_data = []
    for _, row in gdf.iterrows():
        coords = [(point[1], point[0]) for point in row['geometry'].coords]
        line_info = {
            "coords": coords,
            "name": row['NAME'],
            "rate1": float(row['RATE1']),
            "rate2": float(row['RATE2']),
            "line_status": row['Line Status'],
            "voltage": float(row['Voltage']),
        }
        line_data.append(line_info)
    return line_data



# Get transmission line data
transmission_line_data = get_line_data(transmission_line_gdf)


# Function to calculate demand by balancing authority over a range of days
def calculate_demand_by_ba(load_gdf, BAsData, start_date, end_date, ba_name):

    # Initialize a dictionary to store results, with nested dictionaries for each date
    DemandBA = {ba_name: {}}

    # Iterate over each day in the date range
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        
        prev_date = current_date - timedelta(days=1)
        

        # Initialize demand data for the specified BA for the current date
        DemandBA[ba_name][current_date.strftime('%Y-%m-%d')] = {"count": 0, "total_Pl": 0, "total_Ql": 0}


        # Iterate over each load point in the GeoDataFrame
        for idx, point in load_gdf.iterrows():
            point_location = Point(point['geometry'].x, point['geometry'].y)  # Get the geometry point
            # Get the polygon for the specified BA
            ba_data = BAsData.get(ba_name)
            if ba_data:
                polygon = Polygon(ba_data['polygon'])  # Create the polygon from BA data
                if polygon.contains(point_location):  # Check if the point is in the polygon
                    DemandBA[ba_name][current_date.strftime('%Y-%m-%d')]["count"] += 1
                    
                    variable_names = [
                        "Daily Maximum Near-Surface Air Temperature",
                    ]
                    
                    # Get weather data for the current and previous day
                    data = get_weather_data_BA(year, month, day, variable_names, point['geometry'].y, point['geometry'].x, variables=("tasmax",))
                    
                    data_prevday = get_weather_data_BA(prev_date.year, prev_date.month, prev_date.day, variable_names, point['geometry'].y, point['geometry'].x, variables=("tasmax",))
                    
                    # Calculate PL_day and QL_day for the current day
                    PL_day_current = point["PL"]  * (1 + 0.01*(5.33 - 0.067 * point['geometry'].x) * (data['tasmax']['value'] - data_prevday['tasmax']['value']))
                    PL_day_current = max(0, PL_day_current)  # Ensure PL_day_current is not negative
                    
                    QL_day_current = point["QL"]  * (1 + 0.01*(5.33 - 0.067 * point['geometry'].x) * (data['tasmax']['value'] - data_prevday['tasmax']['value']))
                
                    DemandBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_Pl"] += PL_day_current  
                    DemandBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_Ql"] += QL_day_current 

        # Move to the next day
        current_date += timedelta(days=1)

    return DemandBA


# Function to calculate generation by balancing authority
def calculate_gen_by_ba(gen_gdf, BAsData, start_date, end_date, ba_name):
       
    # Initialize a dictionary to store results, with nested dictionaries for each date
    GenBA = {ba_name: {}}
    
    # Iterate over each day in the date range
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        
        prev_date = current_date - timedelta(days=1)
    
        # Initialize demand data for each BA for the current date
        GenBA[ba_name][current_date.strftime('%Y-%m-%d')] = {"count": 0, "total_pgen": 0, "total_qgen": 0, "total_fossil": 0, "total_Renw": 0, "total_Wind": 0, "total_Solar": 0, "total_Hydro": 0, "total_Storage": 0}
    
        # Iterate over each generation point in the GeoDataFrame
        for idx, point in gen_gdf.iterrows():
            point_location = Point(point['geometry'].x, point['geometry'].y)  # Get the geometry point
            # Get the polygon for the specified BA
            ba_data = BAsData.get(ba_name)
            if ba_data:
                polygon = Polygon(ba_data['polygon'])  # Create the polygon from BA data
                if polygon.contains(point_location):  # Check if the point is in the polygon
                    GenBA[ba_name][current_date.strftime('%Y-%m-%d')]["count"] += 1
                    
                    variable_names = [
                        "Daily-Mean Near-Surface Wind Speed",
                        "Daily Maximum Near-Surface Air Temperature",
                    ]
                    
                    # Get weather data for the current and previous day
                    data = get_weather_data_BA(year, month, day, variable_names, point['geometry'].y, point['geometry'].x, variables=("sfcWind", "tasmax"))
                    
                    data_prevday = get_weather_data_BA(prev_date.year, prev_date.month, prev_date.day, variable_names, point['geometry'].y, point['geometry'].x, variables=("sfcWind", "tasmax"))
                    
                    # Calculate Pgen_day and Qgen_day for the current day
                    if point["SubType"] == 'WT-Onshore':
                        V_cutin = 3
                        V_cutout = 25
                        V_rated = 12
                        if data['sfcWind']['value'] < V_cutin or data['sfcWind']['value'] > V_cutout:
                            Pgen_day_current = 0
                        elif V_cutin <= data['sfcWind']['value'] < V_rated:
                            Pgen_day_current = point["PG"]  * ((data['sfcWind']['value'] - V_cutin) / (V_rated - V_cutin))
                        else:
                            Pgen_day_current = point["PG"] 
                        
                        eff = 1
                        Qgen_day_current = point["QG"]
                    elif point["SubType"] in ['SolarPV-Tracking', 'SolarPV-NonTracking']:
                        T_th_PV = 35
                        ro_sf = 0.02
                        eff_no = 0.6
                        
                        if data['tasmax']['value'] <= T_th_PV:
                            eff = 1
                        else:
                            eff = eff_no * (1 - ro_sf * (data['tasmax']['value'] - T_th_PV))
                        
                        Pgen_day_current = point["PG"] * eff
                        Qgen_day_current = point["QG"]
                
                    else:
                        T_th_gen = 40
                        ro_th = 0.031
                        
                        if data['tasmax']['value'] <= T_th_gen:
                            eff = 1
                        else:
                            eff = 1 - ro_th * (data['tasmax']['value'] - T_th_gen)
                    
                        Pgen_day_current = point["PG"] * eff
                        Qgen_day_current = point["QG"] * eff
                        
                    
                   
                    
                    GenBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_pgen"] += Pgen_day_current 
                    GenBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_qgen"] += Qgen_day_current 
                    
                    # Classify the generation type and sum the values accordingly
                    if point["SubType"] in ['Battery Storage', 'Bio-ST']:
                        GenBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_Storage"] += Pgen_day_current
                    if point["SubType"] in ['CCWhole-NatGas-Aero', 'CCWhole-NatGas-Industrial', 'CCWhole-NatGas-SingleShaft', 'CT-NatGas-Aero', 'CT-NatGas-Industrial', 'DC-Intertie', 'ICE-NatGas', 'ST-Coal', 'ST-NatGas', 'ST-Nuclear']:
                        GenBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_fossil"] += Pgen_day_current
                    if point["SubType"] in ['DG-BTM', 'Geo-BinaryCycle', 'WT-Onshore', 'SolarPV-NonTracking', 'SolarPV-Tracking']:
                        GenBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_Renw"] += Pgen_day_current
                    if point["SubType"] in ['Hydro', 'HydroRPS']:
                        GenBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_Hydro"] += Pgen_day_current
                    if point["SubType"] in ['SolarPV-NonTracking', 'SolarPV-Tracking']:
                        GenBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_Solar"] += Pgen_day_current
                    if point["SubType"] in ['WT-Onshore']:
                        GenBA[ba_name][current_date.strftime('%Y-%m-%d')]["total_Wind"] += Pgen_day_current
                        
            # Move to the next day
        current_date += timedelta(days=1)
    return GenBA


# Function to create polygons from BA data
def create_BA_polygons(df):
    BAsData = {}
    for index, row in df.iterrows():
        coords = []
        for coord in row[1:]:  # Assuming first column is the BA name
            if pd.notnull(coord):
                try:
                    lat, lon = map(float, coord.split(','))
                    coords.append((lon, lat))  # Append as (longitude, latitude)
                except ValueError:
                    print(f"Skipping invalid coordinate: {coord}")

        # Only create a polygon if there are valid coordinates
        if coords:
            ba_name = row[0]  # Assuming the first column contains the BA name
            polygon = Polygon(coords)
            BAsData[ba_name] = {"polygon": polygon}

    return BAsData


# Read BA boundary data
df_BA = pd.read_excel('Data/BA_GPS_Data.xlsx', sheet_name='Sheet2')

# Create BA polygons
BAsData = create_BA_polygons(df_BA)

# Global cache for demand and generation by balancing authority
cache = {
    'dates': None,
    'demand_by_ba': None,
    'gen_by_ba': None
}

def get_cache_filename(ba_name, start_date_str, end_date_str):
    return f'cache_{ba_name}_{start_date_str}_{end_date_str}.json'

def load_cache(ba_name, start_date_str, end_date_str):
    cache_filename = get_cache_filename(ba_name, start_date_str, end_date_str)
    if os.path.exists(cache_filename):
        with open(cache_filename, 'r') as cache_file:
            return json.load(cache_file)
    return None

def save_cache(ba_name, start_date_str, end_date_str, demand_by_ba, gen_by_ba):
    cache_filename = get_cache_filename(ba_name, start_date_str, end_date_str)
    cache_data = {
        'dates': (start_date_str, end_date_str),
        'demand_by_ba': demand_by_ba,
        'gen_by_ba': gen_by_ba
    }
    with open(cache_filename, 'w') as cache_file:
        json.dump(cache_data, cache_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map():

    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    
    start_date_str = request.args.get('start_date', '2020-07-21')
    end_date_str = request.args.get('end_date', '2020-07-30')
    num_days = int(request.args.get('num_days', 1))
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    heatmap_index = 0
    # Generate heatmaps for the selected date range
    heatmaps = generate_heatmap(start_date, end_date, heatmap_index)
    
    return render_template('map.html', 
                           heatmaps=heatmaps, 
                           num_days=num_days, 
                           start_date=start_date_str, 
                           end_date=end_date_str,
                           transmission_line_data=transmission_line_data,
                           gen_points_data=gen_points_data, 
                           bus_points_data=bus_points_data,
                           substation_points_data=substation_points_data,
                           load_points_data=load_points_data,
                           date=date_str)


@app.route('/get_heatmap_data', methods=['POST'])
def get_heatmap_data():
    data = request.get_json()
    heatmap_index = data.get('heatmap_index')
    
    start_date_str = data.get('start_date', '2020-07-21')
    end_date_str = data.get('end_date', '2020-07-30')

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    if heatmap_index in [0, 4, 8]:
        # Generate heatmaps for the selected date range
        heatmaps = generate_heatmap(start_date, end_date, heatmap_index)
    else:
        num_days = (end_date - start_date).days + 1    
        # Generate heatmaps for the selected date range
        heatmaps = generate_heatmap_diff(start_date, num_days, heatmap_index)
        
    return jsonify({
        'heatmaps': heatmaps
    })

    
    
@app.route('/update_date', methods=['POST'])
def update_date():
    start_date_str = request.form['start_date']
    end_date_str = request.form['end_date']
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    num_days = (end_date - start_date).days + 1

    return redirect(url_for('map', start_date=start_date_str, end_date=end_date_str, num_days=num_days))


@app.route('/plot', methods=['GET'])
def show_plot():
    try:
        # Fetch type from request parameters
        type = request.args.get('type')
        if type == 'line':
            start_date_str = request.args.get('start_date')
            end_date_str = request.args.get('end_date')
            if not start_date_str or not end_date_str:
               return "Error: Missing start_date or end_date form data line", 400
            
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            # Print start_date and end_date
            print(f"Start Date: {start_date_str}")
            print(f"End Date: {end_date_str}")
            
            num_days = (end_date - start_date).days + 1
            
            year, month, day = start_date.year, start_date.month, start_date.day

            # Fetch latitude, longitudefrom request parameters for start point
            lat_st = float(request.args.get('startLat'))
            lon_st = float(request.args.get('startLon'))
            
            # Fetch latitude, longitudefrom request parameters for end point
            lat_end = float(request.args.get('endLat'))
            lon_end = float(request.args.get('endLon'))
            send_info = float(request.args.get('rate2'))

            data_st = fetch_weather_data_for_range(year, month, day, lat_st, lon_st, num_days, type, send_info)
            data_end = fetch_weather_data_for_range(year, month, day, lat_end, lon_end, num_days, type, send_info)
            data_avg = calculate_average_data(data_st, data_end)

            plot_html, min_values, max_values = create_time_series_plot_line(data_st, data_end, data_avg)

            return render_template('plot.html', plot_html=plot_html, min_values=min_values, max_values=max_values)
                                  
            
        else:
            # Extract parameters from the query string
            start_date_str = request.args.get('start_date')
            end_date_str = request.args.get('end_date')
        
            if not start_date_str or not end_date_str:  
                return "Error: Missing start_date or end_date parameters", 400
        
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            num_days = (end_date - start_date).days + 1

            year, month, day = start_date.year, start_date.month, start_date.day
        
            # Fetch latitude, longitude, and type from request parameters
            lat = float(request.args.get('lat'))
            lon = float(request.args.get('lon'))
 
        
            # Fetch data based on type
            if type == 'load':
                pl = float(request.args.get('pl'))
                ql = float(request.args.get('ql'))
                send_info = (pl, ql)
            elif type == 'gen':
                pgen = float(request.args.get('pgen'))
                qgen = float(request.args.get('qgen'))
                gen_type = request.args.get('gen_type')
                send_info = (pgen, qgen, gen_type)
            elif type in ['substation', 'node']:
                send_info = []
            else:
                return "Error: Unsupported type kkkkkkkkkk", 400

            data = fetch_weather_data_for_range(year, month, day, lat, lon, num_days, type, send_info)

            plot_html, min_values, max_values = create_time_series_plot(data, type)

            return render_template('plot.html', plot_html=plot_html, min_values=min_values, max_values=max_values)
    except Exception as e:
        return f"Error: {e}", 500


@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Fetch type from request parameters
        type = request.args.get('type')
        if type == 'line':
            start_date_str = request.form.get('start_date')
            end_date_str = request.form.get('end_date')
            if not start_date_str or not end_date_str:
               return "Error: Missing start_date or end_date form data", 400
            
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            num_days = (end_date - start_date).days + 1
            
            year, month, day = start_date.year, start_date.month, start_date.day

            # Fetch latitude, longitudefrom request parameters for start point
            lat_st = float(request.args.get('startLat'))
            lon_st = float(request.args.get('startLon'))
            
            # Fetch latitude, longitudefrom request parameters for end point
            lat_end = float(request.args.get('endLat'))
            lon_end = float(request.args.get('endLon'))
            send_info = float(request.args.get('rate2'))

            data_st = fetch_weather_data_for_range(year, month, day, lat_st, lon_st, num_days, type, send_info)
            data_end = fetch_weather_data_for_range(year, month, day, lat_end, lon_end, num_days, type, send_info)
            data_avg = calculate_average_data(data_st, data_end)

            min_values, max_values = calculate_min_max(data_avg)

            summary_table = [
               ('Max Temperature', min_values.get('Max_temp', 'N/A')[0], min_values.get('Max_temp', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Max_temp') else 'N/A'),
               ('Avg Temperature', min_values.get('Avg_temp', 'N/A')[0], min_values.get('Avg_temp', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Avg_temp') else 'N/A'),
               ('Min Temperature', min_values.get('Min_temp', 'N/A')[0], min_values.get('Min_temp', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Min_temp') else 'N/A'),
               ('Relative Humidity', min_values.get('Rel_hum', 'N/A')[0], min_values.get('Rel_hum', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Rel_hum') else 'N/A'),
               ('Specific Humidity', min_values.get('Spec_hum', 'N/A')[0], min_values.get('Spec_hum', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Spec_hum') else 'N/A'),
               ('Longwave Radiation', min_values.get('Long_rad', 'N/A')[0], min_values.get('Long_rad', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Long_rad') else 'N/A'),
               ('Shortwave Radiation', min_values.get('Short_rad', 'N/A')[0], min_values.get('Short_rad', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Short_rad') else 'N/A'),
               ('Precipitation', min_values.get('Perc', 'N/A')[0], min_values.get('Perc', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Perc') else 'N/A'),
               ('Wind Speed', min_values.get('Wind_speed', 'N/A')[0], min_values.get('Wind_speed', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Wind_speed') else 'N/A'),
            ]
            
            plot_html, _, _ = create_time_series_plot_line(data_st, data_end, data_avg)

            return render_template('plot.html', summary_table=summary_table, plot_html=plot_html)
        
        
            
            
        else:
            start_date_str = request.form.get('start_date')
            end_date_str = request.form.get('end_date')

            if not start_date_str or not end_date_str:
               return "Error: Missing start_date or end_date form data", 400

            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            num_days = (end_date - start_date).days + 1

        
            year, month, day = start_date.year, start_date.month, start_date.day

            # Fetch latitude, longitudefrom request parameters
            lat = float(request.args.get('lat'))
            lon = float(request.args.get('lon'))


        
            if lat is None or lon is None:
              return "Error: Latitude or Longitude data is missing for the specified point.", 400

             # Fetch data based on type
            if type == 'load':
               pl = float(request.args.get('pl'))
               ql = float(request.args.get('ql'))
               send_info = (pl, ql)
            elif type == 'gen':
               pgen = float(request.args.get('pgen'))
               qgen = float(request.args.get('qgen'))
               gen_type = request.args.get('gen_type')
               send_info = (pgen, qgen, gen_type)
            elif type in ['substation', 'node']:
               send_info = []
            else:
               return "Error: Unsupported type gggggg", 400

            data = fetch_weather_data_for_range(year, month, day, lat, lon, num_days, type, send_info)
        
            min_values, max_values = calculate_min_max(data)

            summary_table = [
               ('Max Temperature', min_values.get('Max_temp', 'N/A')[0], min_values.get('Max_temp', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Max_temp') else 'N/A'),
               ('Avg Temperature', min_values.get('Avg_temp', 'N/A')[0], min_values.get('Avg_temp', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Avg_temp') else 'N/A'),
               ('Min Temperature', min_values.get('Min_temp', 'N/A')[0], min_values.get('Min_temp', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Min_temp') else 'N/A'),
               ('Relative Humidity', min_values.get('Rel_hum', 'N/A')[0], min_values.get('Rel_hum', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Rel_hum') else 'N/A'),
               ('Specific Humidity', min_values.get('Spec_hum', 'N/A')[0], min_values.get('Spec_hum', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Spec_hum') else 'N/A'),
               ('Longwave Radiation', min_values.get('Long_rad', 'N/A')[0], min_values.get('Long_rad', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Long_rad') else 'N/A'),
               ('Shortwave Radiation', min_values.get('Short_rad', 'N/A')[0], min_values.get('Short_rad', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Short_rad') else 'N/A'),
               ('Precipitation', min_values.get('Perc', 'N/A')[0], min_values.get('Perc', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Perc') else 'N/A'),
               ('Wind Speed', min_values.get('Wind_speed', 'N/A')[0], min_values.get('Wind_speed', 'N/A')[1].strftime('%Y-%m-%d') if min_values.get('Wind_speed') else 'N/A'),
            ]

            plot_html, _, _ = create_time_series_plot(data)

            return render_template('plot.html', summary_table=summary_table, plot_html=plot_html)
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/get_BAs_shape', methods=['POST'])
def get_BAs_shape():
    data = request.get_json()

    start_date_str = data.get('start_date', '2020-07-21')
    end_date_str = data.get('end_date', '2020-07-30')

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400

    if start_date > end_date:
        return jsonify({'error': 'Start date must be before end date'}), 400

    BAs_shape_data = []

    for index, row in df_BA.iterrows():
        area_name = row.iloc[0]

        coords = []
        for coord in row[1:]:
            if pd.notnull(coord):
                try:
                    lat, lon = [float(x) for x in coord.split(',')]
                    coords.append((lon, lat))
                except ValueError:
                    print(f"Skipping invalid coordinate: {coord}")

        if coords:
            polygon = Polygon(coords)
            geojson = gpd.GeoSeries([polygon]).__geo_interface__
            area_color = '#%06X' % random.randint(0, 0xFFFFFF)

            BAs_shape_data.append({
                'geojson': geojson,
                'area_name': area_name,
                'area_color': area_color,
                'centroid': [polygon.centroid.y, polygon.centroid.x],
            })

    return jsonify(BAs_shape_data)
        
@app.route('/get_BAs_data', methods=['POST'])
def get_BAs_data():
    data = request.get_json()

    ba_name = data.get('ba_name')  # Get BA name from POST data
    start_date_str = data.get('start_date', '2020-07-21')
    end_date_str = data.get('end_date', '2020-07-30')

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400

    if start_date > end_date:
        return jsonify({'error': 'Start date must be before end date'}), 400

    # If ba_name is not empty, run the following block
    if ba_name:
        # Check if the cache exists
        cached_data = load_cache(ba_name, start_date_str, end_date_str)
        if cached_data:
            cache['dates'] = cached_data['dates']
            cache['demand_by_ba'] = cached_data['demand_by_ba']
            cache['gen_by_ba'] = cached_data['gen_by_ba']
        else:
            # Recalculate if cache doesn't exist
            cache['dates'] = (start_date_str, end_date_str)
            cache['demand_by_ba'] = calculate_demand_by_ba(load_gdf, BAsData, start_date, end_date, ba_name)
            cache['gen_by_ba'] = calculate_gen_by_ba(gen_gdf, BAsData, start_date, end_date, ba_name)
            
            # Save the calculated data to cache
            save_cache(ba_name, start_date_str, end_date_str, cache['demand_by_ba'], cache['gen_by_ba'])

        demand_by_ba = cache['demand_by_ba']
        gen_by_ba = cache['gen_by_ba']

    else:
        # If ba_name is empty, set default zero values for demand_by_ba and gen_by_ba
        demand_by_ba = {"count": 0,'total_Pl': 0, 'total_Ql': 0}
        gen_by_ba = {"count": 0, "total_pgen": 0, "total_qgen": 0, "total_fossil": 0, "total_Renw": 0, "total_Wind": 0, "total_Solar": 0, "total_Hydro": 0, "total_Storage": 0}

    BAs_data = []

    for index, row in df_BA.iterrows():
        area_name = row.iloc[0]
        file_name = f'Data/EEA-{area_name}.xlsx'

        filtered_events = pd.DataFrame()

        if os.path.exists(file_name):
            df2 = pd.read_excel(file_name)
            filtered_events = df2[df2['Year'].between(start_date.year, end_date.year)]
            column_order = filtered_events.columns.tolist()

        coords = []
        for coord in row[1:]:
            if pd.notnull(coord):
                try:
                    lat, lon = [float(x) for x in coord.split(',')]
                    coords.append((lon, lat))
                except ValueError:
                    print(f"Skipping invalid coordinate: {coord}")

        if coords:
            polygon = Polygon(coords)
            geojson = gpd.GeoSeries([polygon]).__geo_interface__
            area_color = '#%06X' % random.randint(0, 0xFFFFFF)

            BAs_data.append({
                'geojson': geojson,
                'area_name': area_name,
                'area_color': area_color,
                'centroid': [polygon.centroid.y, polygon.centroid.x],
                'Events': filtered_events.to_dict(orient='records'),
                'ColumnOrder': column_order,
                'gen_by_ba': gen_by_ba.get(area_name, {}),
                'demand_by_ba': demand_by_ba.get(area_name, {})
            })

    return jsonify(BAs_data)



if __name__ == '__main__':
    app.run(debug=False)
