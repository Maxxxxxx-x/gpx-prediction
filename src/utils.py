import math
import gpxpy

def get_next_position(lat_deg, lon_deg, ele_m, heading_rad, tilting_rad, speed_mps, time_s):
    R = 6371000  # Earth radius in meters

    # Distance traveled
    distance = speed_mps * time_s

    # Decompose into horizontal and vertical components
    horizontal_distance = distance * math.cos(tilting_rad)
    vertical_distance = distance * math.sin(tilting_rad)

    # Convert current latitude to radians
    lat_rad = math.radians(lat_deg)

    # Delta in radians
    delta_lat_rad = (horizontal_distance * math.cos(heading_rad)) / R
    delta_lon_rad = (horizontal_distance * math.sin(heading_rad)) / (R * math.cos(lat_rad))

    # Convert deltas to degrees
    delta_lat_deg = math.degrees(delta_lat_rad)
    delta_lon_deg = math.degrees(delta_lon_rad)

    # New position
    new_lat = lat_deg + delta_lat_deg
    new_lon = lon_deg + delta_lon_deg
    new_ele = ele_m + vertical_distance

    return new_lat, new_lon, new_ele

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat_rad = math.radians(lat2 - lat1)
    delta_lon_rad = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat_rad / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon_rad / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    return R * c

def get_distance3D(lat1, lon1, ele1, lat2, lon2, ele2):
    horizontal_distance = get_distance(lat1, lon1, lat2, lon2)
    vertical_distance = ele2 - ele1
    return math.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)

def get_nearest_dist(lat, lon, ele, path):
    min_distance = float('inf')
    for point in path:
        distance = get_distance3D(lat, lon, ele, point[0], point[1], point[2])
        if distance < min_distance:
            min_distance = distance
    return min_distance

def parseGPX(data):
    gpx = gpxpy.parse(data)
    result = []
    distance = 0
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if len(result) < 1:
                    result.append((point.latitude, point.longitude, point.elevation))
                    continue
                dist = get_distance3D(result[-1][0], result[-1][1], result[-1][2], point.latitude, point.longitude, point.elevation)
                if dist < 100:
                    distance += dist
                    result.append((point.latitude, point.longitude, point.elevation))
    if distance < 400:
        return None
    return result