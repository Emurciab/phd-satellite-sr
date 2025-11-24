"""Calculate optimal chunk grid size"""
import ee
from Auth.AuthGEE import regions

region = regions[0]
aoi = region['aoi']

# Get AOI bounds
bounds = aoi.bounds().getInfo()['coordinates'][0]
lons = [p[0] for p in bounds]
lats = [p[1] for p in bounds]
min_lon, max_lon = min(lons), max(lons)
min_lat, max_lat = min(lats), max(lats)

# Calculate AOI dimensions
from math import cos, radians
lat_center = (min_lat + max_lat) / 2
meters_per_degree_lon = 111320 * cos(radians(lat_center))
meters_per_degree_lat = 110574

width_m = (max_lon - min_lon) * meters_per_degree_lon
height_m = (max_lat - min_lat) * meters_per_degree_lat

print(f"AOI dimensions:")
print(f"  Width: {width_m:.0f} m")
print(f"  Height: {height_m:.0f} m")

# At 3m resolution
scale = 3
width_px = int(width_m / scale)
height_px = int(height_m / scale)
total_px = width_px * height_px

print(f"\nAt {scale}m resolution:")
print(f"  Width: {width_px} pixels")
print(f"  Height: {height_px} pixels")
print(f"  Total: {total_px:,} pixels")

# GEE limit
limit = 262144
print(f"\nGEE sampleRectangle limit: {limit:,} pixels")

# Calculate minimum grid size needed
import math
min_chunks = math.ceil(total_px / limit)
min_grid = math.ceil(math.sqrt(min_chunks))

print(f"\nMinimum chunks needed: {min_chunks}")
print(f"Minimum grid size: {min_grid}x{min_grid} = {min_grid**2} chunks")

# Test different grid sizes
for grid_size in [6, 7, 8, 9, 10]:
    chunk_width_px = width_px / grid_size
    chunk_height_px = height_px / grid_size
    pixels_per_chunk = chunk_width_px * chunk_height_px
    print(f"\n{grid_size}x{grid_size} grid ({grid_size**2} chunks):")
    print(f"  Pixels per chunk: {pixels_per_chunk:,.0f}")
    if pixels_per_chunk <= limit:
        print(f"  ✓ Under limit!")
    else:
        print(f"  ✗ Exceeds limit by {pixels_per_chunk - limit:,.0f} pixels")
