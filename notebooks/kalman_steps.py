import time
import numpy as np
from helpers import list_gpx_files, load_gpx_file, get_indices
from geo_funcs import calc_geodesic
from matplotlib import pyplot as plt
import pandas as pd
from pykalman import KalmanFilter
from gps_utils import haversine
import datetime as dt
from itertools import cycle

def plot_all_keys(_dict, additional_run_info, x_label="Index", y_label="Speed (km/h)", title="Speed Differences", fr_to=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fib_arr = [2, 3, 5, 7, 9, 11, 14, 17, 20, 23, 26, 30, 34, 38, 42]
    thickness = fib_arr[len(_dict.keys())]
    trans = np.linspace(0.5, 1, len(_dict.keys()))
    for idx, key in enumerate(_dict.keys()):
        if fr_to is not None:
            ax.plot(_dict[key][fr_to[0]:fr_to[1]], label=key, linewidth=thickness, alpha=trans[idx])
        else:
            ax.plot(_dict[key], label=key, linewidth=thickness, alpha=trans[idx])
        thickness -= 2
    if "possible_pause_idx" in additional_run_info.keys():
        for idx in additional_run_info["possible_pause_idx"]:
            _y = 0.5*_dict[key][idx]+0.5*np.max(_dict[key])
            ax.plot([idx, idx], [0, _y], 'k--', linewidth=thickness)
            ax.text(idx, _y, f"{format_time(additional_run_info['pause_durations'][idx])}@{format_time(additional_run_info['activity_time'][idx])}", rotation=90, fontsize=8, 
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

def get_activity_durations(td, min_pause_time, verbose=False):
    # Get the indices of possible pause points
    possible_pause_idx = np.argwhere(td > min_pause_time).flatten()

    # Calculate the activity time so far at each pause point
    activity_time = np.cumsum(td)

    # Adjust activity time at pause points
    activity_time[possible_pause_idx] -= np.cumsum(td)[possible_pause_idx] - np.cumsum(td)[possible_pause_idx - 1]

    # Display the results
    if verbose:
        for pause_idx in possible_pause_idx:
            print(f"Pause point at index {pause_idx}: Activity time so far: {format_time(activity_time[pause_idx])} seconds")
    return possible_pause_idx, activity_time

def step_xx0_find_possible_pause_points(segment, additional_run_info, min_pause_time=10, verbose=False, plot_level=0):
    t = [p.time for p in segment.points]
    training_start_time = format_datetime_with_suffix(t[0])
    if verbose:
        print(f"Training started on {training_start_time}")

    td = np.array([(t[idx]-t[idx-1]).seconds for idx in range(1,len(t))])
    possible_pause_idx, activity_time = get_activity_durations(td, min_pause_time)
    total_activity_time = format_time(activity_time[-1])

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
            "activity_time": activity_time
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

def step_04_round_time(coords):
    coords.index = np.round(coords.index.astype(np.int64), -9).astype('datetime64[ns]')
    return

def step_05_resample(coords, freq='1s'):
    coords = coords.resample(freq).mean()
    return coords

def step_06_get_measurements_from_coords(coords):
    measurements = np.ma.masked_invalid(coords[['lon', 'lat', 'ele']].values)
    return measurements

def step_07_fill_nan_values(coords, measurements, additional_run_info, verbose=0):
    original_coords_idx = np.argwhere(np.asarray(~coords.ele.isnull().array)).squeeze()
    null_elevation_idx = np.argwhere(np.asarray(coords.ele.isnull().array)).squeeze()
    coords = coords.fillna(method='pad')
    filled_coords = coords.iloc[null_elevation_idx]
    if verbose>1:
        print(f"num of filled coords={len(null_elevation_idx)} in {len(coords.ele)}")
    if verbose:
        fig = plt.figure(figsize=(12,12))
        plt.plot(measurements[:,0], measurements[:,1], 'bo', markersize=5, label="original")
        plt.plot(filled_coords['lon'].values, filled_coords['lat'].values, 'r*', markersize=3, label="filled")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=5, fontsize='xx-small')

    if "possible_pause_idx" in additional_run_info.keys():
        for idx in additional_run_info["possible_pause_idx"]:
            _x = measurements[idx,0]
            _y = measurements[idx,1]
            plt.text(_x, _y, f"{format_time(additional_run_info['pause_durations'][idx])}@{format_time(additional_run_info['activity_time'][idx])}", 
                     rotation=0, fontsize=12, 
                     horizontalalignment="right", verticalalignment="center")
                
    return coords, original_coords_idx

def step_08_setup_kalman_filter(measurements):
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
    kf_dict = {
    "F" : np.array([[1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]]),
    "H": np.array([[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]]),
    "R": np.diag([1e-4, 1e-4, 100])**2,
    "Q": Q,
    "initial_state_mean" : np.hstack([measurements[0, :], 3*[0.]]),# works initial_state_covariance = np.diag([1e-3, 1e-3, 100, 1e-4, 1e-4, 1e-4])**2
    "initial_state_covariance": np.diag([1e-4, 1e-4, 50, 1e-6, 1e-6, 1e-6])**2
    }

    kf = KalmanFilter(transition_matrices=kf_dict["F"], 
                    observation_matrices=kf_dict["H"], 
                    observation_covariance=kf_dict["R"],
                    initial_state_mean=kf_dict["initial_state_mean"],
                    initial_state_covariance=kf_dict["initial_state_covariance"],
                    em_vars=['transition_covariance'])
    kf.transition_covariance = kf_dict["Q"]

    return kf, kf_dict

def step_09_apply_kalman(kf, measurements):
    state_means, state_vars = kf.smooth(measurements)
    return state_means, state_vars

def step_10_plot_smoothed_vs_measured(state_means, measurements, additional_run_info):
    fig, axs = plt.subplots(3,1, figsize=(21, 15))
    title_str = ["longitude","Latitude","elevation"]
    for i in range(3):
        # Plot for longitude
        axs[i].plot(state_means[:, i], label="Smoothed", linewidth=5)
        axs[i].plot(measurements[:, i], label="Measurements", linewidth=2)
        if "possible_pause_idx" in additional_run_info.keys():
            unmasked_idx = np.argwhere(~measurements[:,0].mask).squeeze()
            _y_min, _y_mean, _y_max = np.min(measurements[unmasked_idx,i]), np.mean(measurements[unmasked_idx,i]), np.max(measurements[unmasked_idx,i])
            _y_dif = _y_max - _y_min
            _x_max = np.max(unmasked_idx)
            axs[i].plot([0, _x_max], [_y_mean, _y_mean], 'g--', linewidth=2, label="Mean")
            add_min = +0.5

            for idx in additional_run_info["possible_pause_idx"]:
                _uidx = unmasked_idx[idx]
                _x = _uidx
                _y = measurements[_uidx,i]
                _y_norm = (_y - _y_min) / _y_dif
                _y_norm = 0.25 + 0.5*np.abs(_y_norm-0.5)
                _y_norm = add_min*_y_norm
                _y_text = _y_mean+_y_norm*_y_dif
                _y_text = _y_mean-_y_norm*_y_dif if np.abs((_y_text-_y)/_y_dif)<0.2 else _y_text
                _y_text -= add_min*0.2*_y_dif
                add_min *= -1
                axs[i].text(_x, _y_text, f"{format_time(additional_run_info['pause_durations'][idx])}@{format_time(additional_run_info['activity_time'][idx])}", 
                        rotation=20, fontsize=10, horizontalalignment="center", verticalalignment="center")
                axs[i].plot([_x, _x], [_y, _y_text], 'k--', linewidth=2)
        axs[i].set_title(title_str[i])
        axs[i].legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize='small')
        axs[i].tick_params(axis='both', which='major', labelsize='small')


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