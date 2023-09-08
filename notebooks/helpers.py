import os
import gpxpy
import json
import numpy as np

IDENTIFIERS_FILE = "identifiers.json"

def load_identifiers():
    # Load the identifiers and their corresponding file names from the file
    identifiers = {}
    if os.path.exists(IDENTIFIERS_FILE):
        with open(IDENTIFIERS_FILE, "r") as file:
            identifiers = json.load(file)
    print("Identifiers:")
    for idx, identstr in enumerate(identifiers.keys()):
        print(f"{idx}: {identstr} ({identifiers[identstr]['explanation']}))")
    return identifiers

def list_gpx_files():
    gpx_files = []
    for f in os.listdir("../gpx/"):
        if f.endswith(".gpx"):
            full_file = os.path.abspath(os.path.join("../gpx/", f))
            gpx_files.append(full_file)
    print("\n".join(gpx_files))
    return gpx_files

def load_gpx_file(gpx_files, file_name):
    gpx_file_id = [idx for idx, fname in enumerate(gpx_files) if file_name in fname]
    gpx_file_name = gpx_files[gpx_file_id[0]]
    print("Loading file: {}".format(gpx_file_name))
    with open(gpx_file_name) as fh:
        gpx_file = gpxpy.parse(fh)
    return gpx_file

def get_indices(center_idx, threshold, divisible_by):
    lower_limit = (center_idx - threshold) // divisible_by * divisible_by
    upper_limit = ((center_idx + threshold) // divisible_by + 1) * divisible_by
    idx_arr = range(lower_limit, upper_limit)
    return idx_arr

def smooth_vector(input_vector, threshold):
    input_vector = np.array(input_vector)
    sorted_vector = np.sort(input_vector.flatten())
    smoothed_vector = []
    current_group = [sorted_vector[0]]

    for i in range(1, len(sorted_vector)):
        if sorted_vector[i] - sorted_vector[i - 1] <= threshold:
            current_group.append(sorted_vector[i])
        else:
            if len(current_group) == 1:
                smoothed_vector.append(current_group[0])
            else:
                median_value = np.median(current_group)
                smoothed_vector.append(int(median_value))
            current_group = [sorted_vector[i]]

    if len(current_group) == 1:
        smoothed_vector.append(current_group[0])
    else:
        median_value = np.median(current_group)
        smoothed_vector.append(int(median_value))

    return smoothed_vector

def find_direction_change_indices(x, tresh=3):
    vals_mul = x[0:-1]*x[1:]
    idx = np.argwhere(vals_mul<0)
    print("idx:", end=":(")
    for i in idx:
        mode = '?' 
        if (x[i]<0 and x[i+1]>0):
            mode = 'INC' 
        if (x[i]>0 and x[i+1]<0):
            mode = 'dec' 
        print(f"{i}:{mode}",end=");(")
    print(")")

    smoothedidx = smooth_vector(np.squeeze(idx), tresh)
    print(f"smoothed:tresh({tresh}):", end=":(")
    for i in smoothedidx:
        mode = '?' 
        if (x[i]<0 and x[i+1]>0):
            mode = 'INC' 
        if (x[i]>0 and x[i+1]<0):
            mode = 'dec' 
        print(f"{i}:{mode}",end=");(")
    print(")")

    return smoothedidx