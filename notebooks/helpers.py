import os
import gpxpy
import json

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