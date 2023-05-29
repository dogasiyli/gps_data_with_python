import numpy as np
from helpers import list_gpx_files, load_gpx_file
from geo_funcs import calc_geodesic
from matplotlib import pyplot as plt
import pandas as pd

def plot_all_keys(speed_dict, x_label="Index", y_label="Speed (km/h)", title="Speed Differences"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fib_arr = [2, 3, 5, 7, 9, 11, 14, 17, 20, 23, 26, 30, 34, 38, 42]
    thickness = fib_arr[len(speed_dict.keys())]
    for key in speed_dict.keys():
        ax.plot(speed_dict[key], label=key, linewidth=thickness)
        thickness -= 2

    ax.legend(loc='lower center', fontsize='xx-small')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

def step_01_load_data():
    gpx_files = list_gpx_files()
    gpx_obj = load_gpx_file(gpx_files, 2)
    segment = gpx_obj.tracks[0].segments[0]
    return gpx_obj, segment

def step_02_initial_speed(gpx_obj, segment, speed_dict):
    segment.points[0].speed = 0.0
    segment.points[-1].speed = 0.0
    gpx_obj.add_missing_speeds()
    speed = np.array([p.speed for p in segment.points])*3.6
    speed_dict['speed_0_initial'] = speed

    speed_vincenty = np.zeros_like(speed)
    for i in range(1, len(segment.points)):
        x = calc_geodesic(segment.points[i-1],segment.points[i],False)
        speed_vincenty[i] = x['kmh']
    
    speed_dict['speed_0_vincenty'] = speed_vincenty

def step_03_segments_to_coords_pd(segment):
    coords = pd.DataFrame([{'idx': i,
                        'lat': p.latitude, 
                        'lon': p.longitude, 
                        'ele': p.elevation,
                        'speed': p.speed,
                        'time': p.time} for i, p in enumerate(segment.points)])
    coords.set_index('time', inplace=True)
    return coords

def step_04_round_time(coords):
    coords.index = np.round(coords.index.astype(np.int64), -9).astype('datetime64[ns]')
    return

def step_05_resample(coords, freq='1s'):
    coords = coords.resample(freq).mean()
    return coords

def step_06_get_measurements_from_coords(coords):
    measurements = np.ma.masked_invalid(coords[['lon', 'lat', 'ele']].values)
    return measurements