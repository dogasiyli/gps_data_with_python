import time
from turtle import width
import numpy as np
from pyparsing import alphanums
from helpers import list_gpx_files, load_gpx_file, get_indices
from geo_funcs import calc_geodesic
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pykalman import KalmanFilter
from gps_utils import haversine
import seaborn as sns
from adjustText import adjust_text

def print_known_info(file_identifier, wanted_info):
    for k in wanted_info:
        if k in file_identifier.keys():
            print(f"known {k}: {file_identifier[k]}")
        else:
            print(f"unknown {k}")

def plot_all_keys(_dict, additional_run_info, x_label="Index", y_label="Speed (km/h)", title="Speed Differences", fr_to=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fib_arr = [2, 3, 5, 7, 9, 11, 14, 17, 20, 23, 26, 30, 34, 38, 42]
    thickness = fib_arr[len(_dict.keys())]
    trans = np.linspace(0.25, 0.75, len(_dict.keys()))
    for idx, key in enumerate(_dict.keys()):
        if fr_to is not None:
            ax.plot(_dict[key][fr_to[0]:fr_to[1]], label=key, linewidth=thickness, alpha=trans[idx])
        else:
            ax.plot(_dict[key], label=key, linewidth=thickness, alpha=trans[idx])
        thickness -= 2
    if "possible_pause_idx" in additional_run_info.keys():
        for _i in range(len(additional_run_info['pause_df'])):
            idx = additional_run_info["possible_pause_idx"][_i]
            _y = 0.5*_dict[key][idx]+0.5*np.max(_dict[key])
            ax.plot([idx, idx], [0, _y], 'k--', linewidth=thickness)
            ax.text(idx, _y, f"{format_time(additional_run_info['pause_df']['duration'][_i])}@{format_time(additional_run_info['active_df']['tot_activity_time'][_i])}", rotation=90, fontsize=8, 
                         horizontalalignment="center", verticalalignment="bottom")

    ax.legend(bbox_to_anchor=(1.01, 1), borderaxespad=5, fontsize='xx-small')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()

def plot_all_key_difs(_dict, x_label="Index", y_label="Speed (km/h)", title="Speed Differences"):
    thickness = 5
    all_keys = list(_dict.keys())
    key_count = len(all_keys)
    sub_plot_count = (key_count*(key_count-1))//2
    fig, ax = plt.subplots(sub_plot_count,1, figsize=(12, sub_plot_count*3))
    k = 0
    for i in range(len(all_keys)-1):
        for j in range(i+1, len(all_keys)):
            key_i = all_keys[i]
            key_j = all_keys[j]
            _ax = ax if sub_plot_count==1 else ax[k]
            if len(_dict[key_i])==len(_dict[key_j]):
                _plot_vec = abs(_dict[key_i]-_dict[key_j])
                _ax.plot(_plot_vec, linewidth=thickness)
                _ax.plot([0, len(_plot_vec)], [np.mean(_plot_vec), np.mean(_plot_vec)], 'r--', linewidth=thickness)
                _ax.text(0, np.max(_plot_vec), f"mean={np.mean(_plot_vec):.2f}")
                _ax.set_title(f"{key_i}-{key_j}")
                _ax.set_xlabel(x_label)
                _ax.set_ylabel(y_label)
            k += 1
    plt.tight_layout(pad=2.5)
    plt.show()

def plot_time_dif_counts(coords):
    time_delta = np.abs(np.diff(coords.index).astype('timedelta64[s]').astype(int))
    uniqe_time_delta = np.unique(time_delta, return_index=True, return_counts=True)
    ut_df = pd.DataFrame({'time_delta': uniqe_time_delta[0], 'count': uniqe_time_delta[2]})

    ut_df['index'] = 0  # Set all index values to 0

    plt.figure(figsize=(18, 2))
    pivot_df = ut_df.pivot(index="time_delta", columns="index", values="count").T
    sns.heatmap(pivot_df, annot=True, fmt="4.0f", annot_kws={"horizontalalignment": "center"}, cmap='coolwarm')
    plt.xlabel("Time Delta")
    plt.ylabel("Count")
    plt.show()

def step_01_load_data(file_name):
    gpx_files = list_gpx_files()
    gpx_obj = load_gpx_file(gpx_files, file_name)
    print(f"num of tracks={len(gpx_obj.tracks)}")
    segment = gpx_obj.tracks[0].segments[0]
    return gpx_obj, segment

def format_datetime_with_suffix(dt_param):
    day = dt_param.strftime("%d")
    suffix = "th" if 11 <= int(day) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(int(day) % 10, "th")
    formatted_date = dt_param.strftime(f"%d{suffix} of %B %Y at %H:%M%p.")
    return formatted_date

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    formatted_time = ""
    
    if hours > 0:
        formatted_time += f"{hours}:"
    if minutes > 0:
        formatted_time += f"{minutes}'".zfill(3)
    formatted_time += f'{seconds}"'
    
    return formatted_time

def create_active_pause_df(td, min_pause_time):
    # Get the indices of possible pause points
    td = np.array(list(td)+list([min_pause_time+1])).copy()
    possible_pause_idx = np.argwhere(td > min_pause_time).flatten()

    # Calculate the activity time so far at each pause point
    activity_time = np.cumsum(td)

    # Create active_df DataFrame
    active_data = []
    cum_td = np.cumsum(td)
    for i, pause_idx in enumerate(possible_pause_idx):
        start_idx = 0 if i == 0 else possible_pause_idx[i-1]
        end_idx = pause_idx
        duration = cum_td[end_idx-1]- cum_td[start_idx] if i>0 else cum_td[end_idx-1]
        init_second = cum_td[start_idx] if start_idx>0 else 0
        final_second = cum_td[end_idx-1] if end_idx < len(td) else cum_td[-1]
        upuntilnow = duration if i==0 else np.sum(np.array(active_data)[:,3])+duration
        active_data.append([i, start_idx, end_idx-1, duration, init_second, final_second, upuntilnow])

    active_df = pd.DataFrame(active_data, columns=['id', 'first_idx', 'last_idx', 'duration', 'init_second', 'final_second', 'tot_activity_time'])

    # Create pause_df DataFrame
    pause_data = []
    pause_idx = np.array(active_df['first_idx'][1:])
    duration = np.array(active_df['init_second'][1:]) - np.array(active_df['final_second'][:-1])
    init_second = np.array(active_df['final_second'][:-1])
    final_second = np.array(active_df['init_second'][1:])

    pause_data = list(zip(range(len(pause_idx)), pause_idx, duration, init_second, final_second))
    pause_df = pd.DataFrame(pause_data, columns=['id', 'idx', 'duration', 'init_second', 'final_second'])

    return activity_time, possible_pause_idx, active_df, pause_df

'''
td = np.array([4, 4, 3, 5, 2, 1, 2,3,5,2,1,2,3,2,6,2,1,3])
min_pause_time = 4
possible_pause_idx, activity_time, active_df, pause_df = get_activity_durations(td, min_pause_time, verbose=1)
:
Pause point at index 3: Activity time so far: 16" seconds or 11" seconds
Pause point at index 8: Activity time so far: 29" seconds or 19" seconds
Pause point at index 14: Activity time so far: 45" seconds or 29" seconds
Pause point at index 18: Activity time so far: 56" seconds or 35" seconds
acttive df = 
   id  first_idx  last_idx  duration  init_second  final_second  tot_activity_time
0   0          0         2        11            0            11                 11
1   1          3         7         8           16            24                 19
2   2          8        13        10           29            39                 29
3   3         14        17         6           45            51                 35
pause df = 
   id  idx  duration  init_second  final_second
0   0    3         5           11            16
1   1    8         5           24            29
2   2   14         6           39            45
'''
def get_activity_durations(td, min_pause_time, verbose=False):
    # Create active_df and pause_df DataFrames
    activity_time, possible_pause_idx, active_df, pause_df = create_active_pause_df(td, min_pause_time)

    # Display the results
    if verbose:
        for i, pause_idx in enumerate(possible_pause_idx):
            print(f"Pause point at index {pause_idx}: Activity time so far: {format_time(active_df['tot_activity_time'][i])} seconds")

        print(f"acttive df = \n{active_df}")
        print(f"pause df = \n{pause_df}")
    return possible_pause_idx, activity_time, active_df, pause_df

def step_xx0_find_possible_pause_points(segment, additional_run_info, min_pause_time=10, verbose=False, plot_level=0):
    t = [p.time for p in segment.points]
    training_start_time = format_datetime_with_suffix(t[0])
    if verbose:
        print(f"Training started on {training_start_time}")

    td = np.array([(t[idx]-t[idx-1]).seconds for idx in range(1,len(t))])
    possible_pause_idx, activity_time, active_df, pause_df = get_activity_durations(td, min_pause_time, verbose=verbose)
    total_activity_time = format_time(active_df['tot_activity_time'].iloc[-1])

    if plot_level>1:
        tdhist = plt.hist(td, np.unique(td))
        plt.show()
        for idx in range(len(tdhist[0])):
            print(f"{tdhist[0][idx]}:{tdhist[1][idx]}")
    if plot_level>0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(td)
        ax.set_title(f"Training:{training_start_time}\nFor {total_activity_time} -- Possible Pause Points")
        
        _tv = {"l":"left", "c":"center", "r":"right", "t":"top", "b":"bottom"}
        _prev_td = 0
        for idx,_td in enumerate(td):
            if _td>min_pause_time:
                _tdprev = activity_time[idx]-_prev_td
                allign_code = "cb" if _tdprev>120 else "lb"
                ax.text(idx, _td, f"{str(format_time(_td))}@\n{str(format_time(activity_time[idx]))}", rotation=30, fontsize=8, 
                         horizontalalignment=_tv[allign_code[0]], verticalalignment=_tv[allign_code[1]])
                #print(f"idx({idx}), horizontalalignment({_tv[allign_code[0]]}), verticalalignment({_tv[allign_code[1]]})")
                _prev_td = activity_time[idx]
        ax.text(len(td)-1, 2*td[-1], f"total_time@\n{total_activity_time}", rotation=30, fontsize=8, 
                         horizontalalignment=_tv["c"], verticalalignment=_tv["b"])
        
        plt.tight_layout()
        plt.show()

        additional_run_info = {
            "training_start_time": training_start_time,
            "total_activity_time": total_activity_time,
            "pause_durations": td,
            "possible_pause_idx": possible_pause_idx,
            "activity_time": activity_time,
            "active_df": active_df, 
            "pause_df": pause_df
        }

    return additional_run_info

def interpolate_point(prev_point, next_point):
    interpolated_point = {
        'latitude': np.interp(0.5, [0, 1], [prev_point['latitude'], next_point['latitude']]),
        'longitude': np.interp(0.5, [0, 1], [prev_point['longitude'], next_point['longitude']]),
        'elevation': np.interp(0.5, [0, 1], [prev_point['elevation'], next_point['elevation']]),
        'speed': np.interp(0.5, [0, 1], [prev_point['speed'], next_point['speed']]),
        'time': np.interp(0.5, [0, 1], [prev_point['time'], next_point['time']])
    }
    return interpolated_point

def fix_none_in_speed(speed, segment):
    none_indices = np.argwhere(speed==None).squeeze()  # Get indices of None values in speed array
    if len(none_indices) == 0:
        return speed, segment
    print(f"none_indices = {none_indices}")
    segment.points = [p for idx, p in enumerate(segment.points) if idx not in none_indices]
    speed = np.array([p.speed for p in segment.points])
    # for idx in none_indices:
    #     if idx == 0:
    #         # If the first element is None, find the first non-None value after it
    #         next_idx = np.where(speed[idx+1:] is not None)[0][0] + idx + 1
    #         speed[idx] = np.interp(idx, [idx, next_idx], [speed[idx], speed[next_idx]])
    #         segment.points[idx] = interpolate_point(segment.points[idx], segment.points[next_idx])
    #     elif idx == len(speed) - 1:
    #         # If the last element is None, find the last non-None value before it
    #         prev_idx = np.where(speed[:idx] is not None)[0][-1]
    #         speed[idx] = np.interp(idx, [prev_idx, idx], [speed[prev_idx], speed[idx]])
    #         segment.points[idx] = interpolate_point(segment.points[prev_idx], segment.points[idx])
    #     else:
    #         # Find the previous and next non-None values
    #         prev_idx = np.where(speed[:idx] is not None)[0][-1]
    #         next_idx = np.where(speed[idx+1:] is not None)[0][0] + idx + 1
            
    #         # Linearly interpolate the None value
    #         speed[idx] = np.interp(idx, [prev_idx, next_idx], [speed[prev_idx], speed[next_idx]])
    #         segment.points[idx] = interpolate_point(segment.points[prev_idx], segment.points[next_idx])
    return speed, segment

def calc_speed_distance_vincent(segment, verbose=False):
    speed_vincenty = np.zeros(len(segment.points))
    distance_vincenty = np.zeros(len(segment.points))
    for i in range(1, len(segment.points)):
        x = calc_geodesic(segment.points[i-1],segment.points[i],False)
        speed_vincenty[i] = x['kmh']
        distance_vincenty[i] = x['s_geo_len']
    if verbose:
        print(f"In {len(segment.points)} points, speed_vincenty={np.mean(speed_vincenty)} and distance_vincenty={np.sum(distance_vincenty)/1000:4.3f} km")
    return speed_vincenty, distance_vincenty

def step_02_initial_speed(gpx_obj, segment, speed_dict, verbose=False):
    segment.points[0].speed = 0.0
    segment.points[-1].speed = 0.0
    gpx_obj.add_missing_speeds()
    speed = np.array([p.speed for p in segment.points])

    speed, segment = fix_none_in_speed(speed, segment)

    speed *= 3.6
    speed_dict['speed_0_initial'] = speed
    speed_vincenty, distance_vincenty = calc_speed_distance_vincent(segment, verbose)
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

def add_index_noise(_df, min_noise=200, max_noise=300):
    # Generate random noise for each element in the index
    min_noise = pd.Timedelta(milliseconds=min_noise)
    max_noise = pd.Timedelta(milliseconds=max_noise)
    noise = np.random.uniform(min_noise.total_seconds(), max_noise.total_seconds(), size=len(_df))
    noise = pd.to_timedelta(noise, unit='s')

    # Add the noise to the index
    _df.index = _df.index + noise
    _df.index.name = 'time'

    return _df

def step_04_round_time(_df, frequency='1s'):
    rounded_index = _df.index.ceil(frequency)
    _df.index = rounded_index
    _df.index.name = 'time'
    return _df

def step_05_resample(coords, freq='1s'):
    coords = coords.resample(freq).mean()
    return coords

def step_06_get_measurements_from_coords(coords, additional_run_info):
    measurements = []
    measurements_df_info = pd.DataFrame(columns=['id', 'start_idx', 'end_idx', 'start_time', 'end_time',
                                    'unmasked_count', 'masked_count', 'total_count', 'active_seconds'])

    all_measurements = np.ma.masked_invalid(coords[['lon']].values)
    unmasked_indices = np.nonzero(~all_measurements.mask)[0]

    start_idx = 0
    active_df = additional_run_info['active_df']
    block_cnt = len(active_df)
    for i in range(block_cnt):
        fr = active_df['first_idx'][i] + int(i!=0)
        to = min(active_df['last_idx'][i]+1, len(unmasked_indices)-1)
        start_idx, end_idx = unmasked_indices[fr], unmasked_indices[to]
        measurement = np.ma.masked_invalid(coords[['lon', 'lat', 'ele']].values[start_idx:end_idx])
        measurements.append(measurement)

        start_time = coords.index[start_idx]
        end_time = coords.index[end_idx]
        unmasked_count = np.count_nonzero(~measurement.mask)//3
        masked_count = np.count_nonzero(measurement.mask)//3
        total_count = unmasked_count + masked_count

        measurements_df_info.loc[i] = [i, start_idx, end_idx, start_time, end_time, unmasked_count,
                          masked_count, total_count, active_df['duration'][i]]

    return measurements, measurements_df_info

def step_07_fill_nan_values(coords, measurements, measurements_df_info, verbose=0):
    original_coords_idx = np.argwhere(np.asarray(~coords.ele.isnull().array)).squeeze()
    null_elevation_idx = np.argwhere(np.asarray(coords.ele.isnull().array)).squeeze()
    coords = coords.fillna(method='pad')
    filled_coords = coords.iloc[null_elevation_idx]

    # cmap = cm.get_cmap('cool')  # Defined colormap
    # # Generate a color array based on the length of X using the defined colormap
    # mcolors = cmap(np.linspace(0, 1, len(measurements)))
    mcolors = np.random.rand(len(measurements), 4) * 0.75 + 0.25
    mcolors[:, 3] = 1.0

    if verbose>1:
        print(f"num of filled coords={len(null_elevation_idx)} in {len(coords.ele)}")
    if verbose:
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='purple')
        ax.set_facecolor('black')
        for i_block, m in enumerate(measurements):
            ip = (i_block / len(measurements))
            um = np.ma.compress_rows(m)
            ax.plot(um[:, 0], um[:, 1], color=mcolors[i_block], alpha=0.75+0.25*ip, linestyle='-', linewidth=3+10*ip)
        ax.plot(filled_coords['lon'].values, filled_coords['lat'].values, '.', color='yellow', markersize=4, label="filled")
        plt.xticks(rotation=45)
        lg = plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=5, fontsize='xx-small', facecolor='black')
        for t in lg.get_texts():
            t.set_color('white')  # Set the desired text color

    if measurements_df_info is not None:
        labels = []
        for i_block in range(len(measurements)):
            m = measurements[i_block]
            active_time = format_time(measurements_df_info['active_seconds'][i_block])
            _x = np.nanmean(m[:,0])
            _y = np.nanmean(m[:,1])
            labels.append(plt.text(_x, _y, f"{active_time}@{i_block+1:02d}", color=mcolors[i_block]))
        adjust_text(labels)
                
    return coords, original_coords_idx

def step_08_setup_kalman_filter(measurements, time_interval_in_seconds):
    Q = np.array([[  3.17720723e-09,  -1.56389148e-09,  -2.41793770e-07,
                      2.29258935e-09,  -3.17260647e-09,  -2.89201471e-07],
                  [  1.56687815e-09,   3.16555076e-09,   1.19734906e-07,
                      3.17314157e-09,   2.27469595e-09,  -2.11189940e-08],
                  [ -5.13624053e-08,   2.60171362e-07,   4.62632068e-01,
                      1.00082746e-07,   2.81568920e-07,   6.99461902e-05],
                  [  2.98805710e-09,  -8.62315114e-10,  -1.90678253e-07,
                      5.58468140e-09,  -5.46272629e-09,  -5.75557899e-07],
                  [  8.66285671e-10,   2.97046913e-09,   1.54584155e-07,
                      5.46269262e-09,   5.55161528e-09,   5.67122163e-08],
                  [ -9.24540217e-08,   2.09822077e-07,   7.65126136e-05,
                      4.58344911e-08,   5.74790902e-07,   3.89895992e-04]])
    Q = 0.5*(Q + Q.T) # assure symmetry
    # Careful here, expectation maximation takes several hours!
    # kf = kf.em(measurements, n_iter=1000)
    # or just run this instead of the one above (it is the same result)
    t = time_interval_in_seconds
    kf_dict = {
    "F" : np.array([[1, 0, 0, t, 0, 0],
                    [0, 1, 0, 0, t, 0],
                    [0, 0, 1, 0, 0, t],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]]),
    "H": np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0]]),
    "R": np.diag([1e-4, 1e-4, 100])**2,
    "Q": Q,
    "initial_state_covariance": np.diag([1e-4, 1e-4, 50, 1e-6, 1e-6, 1e-6])**2
    }

    kf_list = []
    for m in measurements:
        kf = KalmanFilter(transition_matrices=kf_dict["F"].copy(), 
                        observation_matrices=kf_dict["H"].copy(), 
                        observation_covariance=kf_dict["R"].copy(), 
                        initial_state_mean=np.hstack([m[0, :], 3*[0.]]), # works initial_state_covariance = np.diag([1e-3, 1e-3, 100, 1e-4, 1e-4, 1e-4])**2
                        initial_state_covariance=kf_dict["initial_state_covariance"].copy(), 
                        em_vars=['transition_covariance'].copy())
        kf.transition_covariance = kf_dict["Q"].copy()
        kf_list.append(kf)

    return kf_list, kf_dict

def step_09_apply_kalman(kf_list, measurements):
    state_means, state_vars = [], []
    for i, (kf, m) in enumerate(zip(kf_list, measurements)):
        _state_means, _state_vars = kf.smooth(m)
        state_means.append(_state_means)
        state_vars.append(_state_vars)
    return state_means, state_vars

def get_measurement_limits(measurements):
    min_values = np.empty((len(measurements), measurements[0].shape[1]))
    max_values = np.empty((len(measurements), measurements[0].shape[1]))
    for i, arr in enumerate(measurements):
        min_values[i,:] = np.array(np.nanmin(arr, axis=0))  # Calculate the minimum values for each dimension
        max_values[i,:] = np.array(np.nanmax(arr, axis=0)) # Calculate the maximum values for each dimension
    return np.min(min_values, axis=0), np.max(max_values, axis=0)

def step_10_plot_smoothed_vs_measured(state_means, measurements, measurements_df_info):
    fig, axs = plt.subplots(3,1, figsize=(21, 18))
    title_str = ["longitude","Latitude","elevation"]
    mcolors = np.random.rand(len(measurements), 4) * 0.75 + 0.25
    mcolors[:, 3] = 1.0
    min_values, max_values = get_measurement_limits(measurements)
    for i_var in range(3):
        # Plot for longitude
        #labels = []
        _x_add = 0
        for j_block, (s, m) in enumerate(zip(state_means, measurements)):
            x_fr, x_to = _x_add, _x_add+len(m)
            axs[i_var].plot(range(x_fr, x_to), s[:, i_var], color='white', alpha=0.5, linewidth=10)
            label_str=f"block {j_block+1}" if measurements_df_info is None else f"b({1+j_block:02d})-{format_time(measurements_df_info['active_seconds'][j_block])}" 
            u_idx = np.argwhere(~m[:, i_var].mask).squeeze()
            axs[i_var].plot(_x_add + u_idx, m[u_idx, i_var], 'o', color=mcolors[j_block], alpha=1.0, markersize=3, label=label_str)
            axs[i_var].set_facecolor('black')
            axs[i_var].grid(False)
            
            if measurements_df_info is not None:
                unmasked_idx = np.argwhere(~m[:, i_var].mask).squeeze()
                unmasked_x = unmasked_idx + x_fr
                unmasked_y = m[unmasked_idx, i_var]
                _x_median = np.nanmedian(unmasked_x)
                _y_median = np.nanmedian(unmasked_y)
                # vertical line from zero to the block mean
                axs[i_var].plot([_x_median, _x_median], [min_values[i_var], _y_median], 'g--', linewidth=2)
                # text of active time and block number 
                active_time = format_time(measurements_df_info['active_seconds'][j_block])
                axs[i_var].text(_x_median, _y_median, f"{active_time}@{j_block+1:02d}", color=mcolors[j_block], fontsize='small',
                                horizontalalignment='center', verticalalignment='bottom', rotation=90)
            _x_add += len(m)
        #adjust_text(labels)                
        axs[i_var].set_title(title_str[i_var])
        axs[i_var].legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize='small')
        axs[i_var].tick_params(axis='both', which='major', labelsize='small')

    plt.tight_layout(pad=2.5)
    # Move the legends to the right of the plot
    plt.show()

def step_11_15_22_update_coords(state_means, coords, segment, original_coords_idx):
    coords.iloc[:, [2,1,3]] = state_means[:,:3]
    orig_coords = coords.iloc[original_coords_idx].set_index('idx')
    for i, p in enumerate(segment.points):
        p.speed = None
        p.elevation = orig_coords.at[float(i),'ele']
        p.longitude = orig_coords.at[float(i),'lon']
        p.latitude = orig_coords.at[float(i),'lat']
    return coords, orig_coords

def step_12_16_23_add_missing_speeds(gpx_obj, segment, speed_dict, sdict_key, plot_after_add=False):
    segment.points[0].speed = 0.0
    segment.points[-1].speed = 0.0
    gpx_obj.add_missing_speeds()

    speed = np.array([p.speed for p in segment.points])*3.6
    speed_dict[sdict_key] = speed
    if plot_after_add:
        plt.plot(speed)
    return speed

def step_13_get_high_speed_block(speed, measurements, segment, plot_result=False, verbose=False):
    fig, axs = plt.subplots(2,1, figsize=(12, 12))
    sdifs = speed[1:]-speed[:-1]
    max_dif_at = np.argsort(-sdifs)
    high_speed_block = get_indices(max_dif_at[0], 10, 10)
    if verbose:
        print(max_dif_at[:10])
        print(sdifs[max_dif_at[:10]])
        print(max_dif_at[1])
        print(min(high_speed_block), max(high_speed_block))
    if plot_result:
        axs[0].plot(measurements[high_speed_block,0], measurements[high_speed_block,1], 'o', alpha=0.5)
        axs[0].set_title(f"High speed block between {min(high_speed_block)} and {max(high_speed_block)} \n {len(high_speed_block)} points")
        axs[0].set_xlabel("Longitude")
        axs[0].set_ylabel("Latitude")

        _s = [p.speed for p in segment.points[min(high_speed_block):max(high_speed_block)]]
        axs[1].scatter(range(len(_s)), _s, alpha=0.5)
        axs[1].set_title(f"speed is in km/h \n {len(_s)} points")

    plt.tight_layout(pad=2.5)
    return high_speed_block

def step_14_find_bad_readings_from_variance(measurements, state_vars, verbose=False):
    bad_readings = np.argsort(np.trace(state_vars[:,:2,:2], axis1=1, axis2=2))[:-20:-1]
    '''
    The line above performs the following step by step:
    1. `state_vars[:,:2,:2]` selects a subset of the `state_vars` array. It takes all rows (`:`) and the first two columns (`:2`) of the first two dimensions (`:2`). This selection is a 3-dimensional array.
    2. `np.trace(..., axis1=1, axis2=2)` calculates the trace of each 2x2 matrix along the specified axes. The `axis1=1` and `axis2=2` arguments indicate that the trace should be calculated along the second and third axes, respectively. This results in a 1-dimensional array with the trace values.
    3. `np.argsort(...)` returns the indices that would sort the array in ascending order. In this case, it sorts the array of trace values in ascending order and returns the corresponding indices.
    4. `[:-20:-1]` selects the last 19 elements in reverse order using slicing. The `[:-20]` part removes the first 20 elements from the sorted indices, and `[::-1]` reverses the order to obtain the top 19 indices.
    5. The resulting indices are stored in the `bad_readings` variable, which represents the indices of the "bad readings" based on some criterion (likely related to the trace values of the `state_vars` array).
    In summary, the line of code identifies the indices of the 19 largest trace values in the specified subset of `state_vars`. These indices are stored in the `bad_readings` variable, indicating the positions of the "bad readings" based on the given criterion.
    '''
    bad_readings = np.array([idx for idx in range(measurements.shape[0]) if np.min(np.abs(bad_readings - idx)) <= 5])
    '''
    the second line refines the bad_readings array generated in the first line 
    by applying an additional condition based on the minimum absolute difference 
    between the elements in bad_readings and the range of indices from 0 to measurements.shape[0] - 1. 
    This allows for further filtering of the "bad readings" based on the updated criterion.
    '''
    if verbose:
        print(f"there are {len(measurements.mask)} of mask vars.")
        print(f"number of true elements in the mask(shape={np.shape(measurements.mask)} with {np.prod(np.shape(measurements.mask))} elements) are {np.sum(measurements.mask)}")
        print(f"there are {len(bad_readings)} bad readings")
        print(f"bad readings where the mask was false count is: {np.sum(measurements.mask[bad_readings,:])//3}")
    measurements.mask[bad_readings, :] = True 
    return measurements, bad_readings

def step_17_21_calc_speed_with_corrected_positions(state_means, speed_dict, sdict_key, measurements, high_speed_block, verbose=False, plot_result=2):
    # calculate the speed directly on our array
    speed = [3.6*haversine(state_means[i,1::-1], state_means[i+1,1::-1]) for i in np.arange(state_means.shape[0]-1)]
    speed_dict[sdict_key] = np.array(speed)
    # the line returns the indices of the 9 largest elements in the speed array, 
    # with the indices arranged in descending order based on the corresponding element values. 
    # This can be useful for identifying the highest values or top elements in the speed array.
    if verbose or plot_result:
        largest_speed_idx = np.argsort(speed)[:-10:-1]
    if plot_result==2:
        fig, axs = plt.subplots(2,1, figsize=(12, 12))
    elif plot_result==1:
        fig, axs = plt.subplots(1,1, figsize=(12, 6))
        axs = [axs]

    axs[0].plot(speed, label="speed")
    axs[0].plot(speed[min(high_speed_block):max(high_speed_block)], '.', label="speed in high speed block")
    axs[0].plot(largest_speed_idx, [speed[i] for i in largest_speed_idx], 'bo', label="largest speed", markersize=10)
    axs[0].set_title(f"{len(speed)} speed points")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("speed is in km/h")
    axs[0].legend(bbox_to_anchor=(1.05, 1), borderaxespad=5, fontsize='xx-small')

    if plot_result==2:
        print(f"there are {len(measurements.mask)} of mask vars.")
        print(f"number of true elements in the mask(shape={np.shape(measurements.mask)} with {np.prod(np.shape(measurements.mask))} elements) are {np.sum(measurements.mask)}")
        masked_idx = np.where(measurements.mask[:,0] == True)[0]
        unmasked_idx = np.where(measurements.mask[:,0] == False)[0]
        print(f"number of masked elements={len(masked_idx)} unmasked={len(unmasked_idx)}")
        try:
            axs[1].plot(measurements.data[masked_idx][:,0], measurements.data[masked_idx][:,1], 'ro', alpha=0.5, markersize=10, label="masked")
        except:
            pass
        axs[1].plot(measurements.data[unmasked_idx][:,0], measurements.data[unmasked_idx][:,1], 'b.', markersize=2, label="unmasked")
        axs[1].plot(measurements[high_speed_block,0], measurements[high_speed_block,1], 'yo', markersize=10, alpha=0.5, label="high speed block")
        axs[1].set_title(f"High speed block between {min(high_speed_block)} and {max(high_speed_block)} \n {len(high_speed_block)} points")
        axs[1].set_xlabel("Longitude")
        axs[1].set_ylabel("Latitude")
        axs[1].legend(bbox_to_anchor=(1.05, 1), borderaxespad=5, fontsize='xx-small')
        plt.tight_layout(pad=2.5)

def step_18_check_for_outliers_by_strong_accelerations(speed, measurements, plot_result=True):
    # we check for strong accelerations/deaccelarations
    acc = np.gradient(speed)
    outliers_idx = np.argsort(np.abs(acc))[:-40:-1]
    if plot_result:
        fig, axs = plt.subplots(2,1, figsize=(12, 12))
        axs[0].plot(speed, label="speed")
        axs[0].plot(outliers_idx, [speed[i] for i in outliers_idx], 'bo', label="largest accelerations", markersize=10)
        axs[0].set_title(f"{len(speed)} speed points")
        axs[0].set_xlabel("time")
        axs[0].set_ylabel("speed is in km/h")
        axs[0].legend(bbox_to_anchor=(1.05, 1), borderaxespad=5, fontsize='xx-small')

        axs[1].plot(measurements[:,0], measurements[:,1], '.', markersize=1, label="all gps info")
        axs[1].plot(measurements[outliers_idx,0], measurements[outliers_idx,1], 'o', markersize=5, alpha=0.5, label="outliers")
        axs[1].set_title(f"Largest acceleration points \n {len(outliers_idx)} points")
        axs[1].set_xlabel("Longitude")
        axs[1].set_ylabel("Latitude")
        axs[1].legend(bbox_to_anchor=(1.05, 1), borderaxespad=5, fontsize='xx-small')
    plt.tight_layout(pad=2.5)
    return outliers_idx

def step_19_mask_around_outliers(kf, measurements, outliers_idx, neighbour_cnt):
    outliers_idx_new = np.array([idx for idx in range(measurements.shape[0]) if np.min(np.abs(outliers_idx - idx)) <= neighbour_cnt])
    measurements.mask[outliers_idx_new] = True
    state_means, state_vars = kf.smooth(measurements)
    return state_means, state_vars, measurements

def step_20_apply_kalman(state_means, state_vars, kf, n_times):
    durations = []
    # we smooth several times
    print(f"Iterations in sec:", end="")
    for idx in range(n_times):
        start_time = time.time()
        state_means, state_vars = kf.smooth(state_means[:,:3])
        end_time = time.time()

        duration = end_time - start_time
        durations.append(duration)
        print(f"<{idx}:{duration:4.2f}>", end="")
    print("\n")

    min_duration = np.min(durations)
    mean_duration = np.mean(durations)
    max_duration = np.max(durations)

    print(f"Durations: min({min_duration}), mean({mean_duration}), max({max_duration})")
    return state_means, state_vars