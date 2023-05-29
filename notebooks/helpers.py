import os
import gpxpy



def list_gpx_files():
    gpx_files = []
    for f in os.listdir("../gpx/"):
        if f.endswith(".gpx"):
            full_file = os.path.abspath(os.path.join("../gpx/", f))
            gpx_files.append(full_file)
    print("\n".join(gpx_files))
    return gpx_files

def load_gpx_file(gpx_files, gpx_file_id):
    gpx_file_name = gpx_files[gpx_file_id]
    with open(gpx_file_name) as fh:
        gpx_file = gpxpy.parse(fh)
    return gpx_file

def get_indices(center_idx, threshold, divisible_by):
    lower_limit = (center_idx - threshold) // divisible_by * divisible_by
    upper_limit = ((center_idx + threshold) // divisible_by + 1) * divisible_by
    idx_arr = range(lower_limit, upper_limit)
    return idx_arr