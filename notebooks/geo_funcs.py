import math
from datetime import datetime

def calc_pace_from_kmh(kmh, verbose=False):
    # Calculate pace in minutes per kilometer
    pace = 60 / kmh
    # Convert pace to minutes and seconds
    minutes = int(pace)
    seconds = int((pace - minutes) * 60)
    if verbose:
        print("Pace:", minutes, "minutes", seconds, "seconds per kilometer")
    return pace

def calc_run_params(dist_in_meters, time_in_seconds, verbose=False):
    # Calculate meters per second
    mps = dist_in_meters / time_in_seconds
    # Calculate kilometers per hour
    kmh = mps * 3.6
    if verbose:
        print("Kilometers per hour:", kmh)
    pace = calc_pace_from_kmh(kmh, verbose=verbose)
    return pace, kmh

def deg_dms(direction_in_degrees):
    #https://www.calculatorsoup.com/calculators/conversions/convert-decimal-degrees-to-degrees-minutes-seconds.php
    degrees = int(direction_in_degrees)
    minutes_decimal = (direction_in_degrees - degrees) * 60
    minutes = int(minutes_decimal)
    seconds = (minutes_decimal - minutes) * 60
    return f"{degrees} degrees {minutes} minutes {seconds:4.2f} seconds"

# https://www.movable-type.co.uk/scripts/latlong-vincenty.html
# Distance/bearing between two points (inverse solution)
# Distance/bearing between two points (inverse solution)
# @ https://www.movable-type.co.uk/scripts/latlong-vincenty.html
def calc_geodesic(p1, p2, verbose=False):
    try:
        timestamp1 = p1['timestamp']
        timestamp2 = p2['timestamp']
        time_diff = (timestamp2 - timestamp1)/1000
        φ1 = math.radians(p1['coords']['latitude'])
        φ2 = math.radians(p2['coords']['latitude'])
        λ1 = math.radians(p1['coords']['longitude'])
        λ2 = math.radians(p2['coords']['longitude'])
    except:
        t1 = p1.time
        t2 = p2.time
        time_diff = (t2-t1).total_seconds()
        φ1 = math.radians(p1.latitude)
        φ2 = math.radians(p2.latitude)
        λ1 = math.radians(p1.longitude)
        λ2 = math.radians(p2.longitude)        

    if verbose:
        print(f"Time Difference: {time_diff} seconds")

    a = 6378137.0  # Semi-major axis of the Earth
    b = 6356752.314245  # Semi-minor axis of the Earth
    f = (a - b) / a  # Flattening factor

    L = λ2 - λ1 # U = reduced latitude, defined by tan U = (1-f)·tanφ.
    if verbose:
        print(f"L(difference in longitude)={L}")
    tanU1 = (1 - f) * math.tan(φ1)
    tanU2 = (1 - f) * math.tan(φ2)
    cosU1 = 1 / math.sqrt((1 + tanU1 * tanU1))
    cosU2 = 1 / math.sqrt((1 + tanU2 * tanU2))
    sinU1 = tanU1 * cosU1
    sinU2 = tanU2 * cosU2

    # λ = difference in longitude on an auxiliary sphere
    λ, sinλ, cosλ = L, None, None
    if verbose:
        print(f"λ<init=L>(difference in longitude on an auxiliary sphere)={λ}")
    # σ = angular distance P₁ P₂ on the sphere
    σ, sinσ, cosσ = None, None, None
    if verbose:
        print(f"σ(angular distance P₁ P₂ on the sphere)={σ}")
    # σₘ = angular distance on the sphere from the equator to the midpoint of the line
    cos2σₘ = None
    # α = azimuth of the geodesic at the equator
    cosSqα = None

    λʹ = None
    while True:
        sinλ = math.sin(λ)
        cosλ = math.cos(λ)
        sinSqσ = (cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2
        sinσ = math.sqrt(sinSqσ)
        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = math.atan2(sinσ, cosσ)
        sinα = cosU1 * cosU2 * sinλ / sinσ
        cosSqα = 1 - sinα * sinα
        cos2σₘ = cosσ - 2 * sinU1 * sinU2 / cosSqα
        C = f / 16 * cosSqα * (4 + f * (4 - 3 * cosSqα))
        λʹ = λ
        λ = L + (1 - C) * f * sinα * (σ + C * sinσ * (cos2σₘ + C * cosσ * (-1 + 2 * cos2σₘ * cos2σₘ)))
        if math.fabs(λ - λʹ) <= 1e-12:
            break

    uSq = cosSqα * (a * a - b * b) / (b * b)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    Δσ = B * sinσ * (cos2σₘ + B / 4 * (cosσ * (-1 + 2 * cos2σₘ * cos2σₘ) - B / 6 * cos2σₘ * (-3 + 4 * sinσ * sinσ) * (-3 + 4 * cos2σₘ * cos2σₘ)))

    # s_geo_len = length of the geodesic
    s_geo_len = b * A * (σ - Δσ)
    if verbose:
        print(f"s_geo_len(length of the geodesic)={s_geo_len}")
    # α1 =initial bearing
    α1 = math.atan2(cosU2 * sinλ, cosU1 * sinU2 - sinU1 * cosU2 * cosλ)  
    if verbose:
        print(f"α1(initial bearing)={α1} radians")
    # α2 = final bearing
    α2 = math.atan2(cosU1 * sinλ, -sinU1 * cosU2 + cosU1 * sinU2 * cosλ)  
    if verbose:
        print(f"α2(final bearing)={α2} radians")

    α1 = math.degrees(α1)
    α2 = math.degrees(α2)
    if verbose:
        print(f"s(length of the geodesic)={s_geo_len:.3f} meters")
        print(f"α1(initial bearing)={deg_dms(α1)}")
        print(f"α2(final bearing)={deg_dms(α2)}")

    pace, kmh = calc_run_params(s_geo_len, time_diff)

    # s_geo_len is the geodesic distance along the surface of the ellipsoid (in the same units as a & b)
    # α1 is the initial bearing, or forward azimuth
    # α2 is the final bearing (in direction p1→p2)

    results = {
        's_geo_len': s_geo_len,
        'α1': α1,
        'α2': α2,
        'pace': pace,
        'kmh': kmh
    }
    return results

# https://www.movable-type.co.uk/scripts/latlong-vincenty.html
# Destination given distance & bearing from start point (direct solution)
def calc_dif_geo(datapoint1, geodesic_results):
    s_geo_len, α1, α2 = geodesic_results['s_geo_len'],geodesic_results['α1'],geodesic_results['α2']

    φ1 = math.radians(datapoint1['coords']['latitude'])
    λ1 = math.radians(datapoint1['coords']['longitude'])
    a = 6378137.0  # Semi-major axis of the Earth
    b = 6356752.314245  # Semi-minor axis of the Earth
    f = (a - b) / a  # Flattening factor

    sinα1 = math.sin(α1)
    cosα1 = math.cos(α1)
    tanU1 = (1 - f) * math.tan(φ1)
    cosU1 = 1 / math.sqrt((1 + tanU1 * tanU1))
    sinU1 = tanU1 * cosU1
    # σ1 = angular distance on the sphere from the equator to P1
    σ1 = math.atan2(tanU1, cosα1)
    # α = azimuth of the geodesic at the equator
    sinα = cosU1 * sinα1
    cosSqα = 1 - sinα * sinα

    uSq = cosSqα * (a * a - b * b) / (b * b)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))

    # σ = angular distance P₁ P₂ on the sphere
    σ, sinσ, cosσ = s_geo_len/(b*A), None, None
    # σₘ = angular distance on the sphere from the equator to the midpoint of the line
    cos2σₘ = None

    σʹ = None
    while True:
        cos2σₘ = math.cos(2 * σ1 + σ)
        sinσ = math.sin(σ)
        cosσ = math.cos(σ)
        Δσ = B * sinσ * (cos2σₘ + B / 4 * (cosσ * (-1 + 2 * cos2σₘ * cos2σₘ) - B / 6 * cos2σₘ * (-3 + 4 * sinσ * sinσ) * (-3 + 4 * cos2σₘ * cos2σₘ)))
        σʹ = σ
        σ = s_geo_len / (b * A) + Δσ
        if math.fabs(σ - σʹ) <= 1e-12:
            break

    x = sinU1 * sinσ - cosU1 * cosσ * cosα1
    φ2 = math.atan2(sinU1 * cosσ + cosU1 * sinσ * cosα1, (1 - f) * math.sqrt(sinα * sinα + x * x))
    λ = math.atan2(sinσ * sinα1, cosU1 * cosσ - sinU1 * sinσ * cosα1)
    C = f / 16 * cosSqα * (4 + f * (4 - 3 * cosSqα))
    L = λ - (1 - C) * f * sinα * (σ + C * sinσ * (cos2σₘ + C * cosσ * (-1 + 2 * cos2σₘ * cos2σₘ)))
    λ2 = λ1 + L

    α2 = math.atan2(sinα, -x)  # final bearing

    # φ2, λ2 is destination point
    # α2 is final bearing (in direction p1→p2)
    
    return φ2, λ2, math.degrees(α2)