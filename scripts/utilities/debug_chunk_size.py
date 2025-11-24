"""Debug: Check chunk dimensions"""
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
from Auth.AuthGEE import find_pairs, regions, PLANET_COLLECTION_ID

region = regions[0]
pairs = find_pairs(region['aoi'], region['start'], region['end'], PLANET_COLLECTION_ID)
planet_collection = pairs.get('planet_collection')
matched = pairs.get('matched_pairs', [])

first = matched[0]
planet_img = planet_collection.filter(ee.Filter.eq('id_pref', first['planet_id'])).first()

# Get AOI bounds
aoi = region['aoi']
bounds = aoi.bounds().getInfo()['coordinates'][0]
lons = [p[0] for p in bounds]
lats = [p[1] for p in bounds]
min_lon, max_lon = min(lons), max(lons)
min_lat, max_lat = min(lats), max(lats)

print(f"AOI size: {max_lon - min_lon:.6f} × {max_lat - min_lat:.6f} degrees")

# At 3m resolution near equator (~111km per degree)
width_m = (max_lon - min_lon) * 111000 * np.cos(np.radians((min_lat + max_lat)/2))
height_m = (max_lat - min_lat) * 111000
print(f"AOI size: {width_m:.0f}m × {height_m:.0f}m")
print(f"Expected pixels at 3m: {width_m/3:.0f} × {height_m/3:.0f}")

# Test single chunk download
lon_step = (max_lon - min_lon) / 3
lat_step = (max_lat - min_lat) / 3

chunk_geom = ee.Geometry.Rectangle([
    min_lon, min_lat,
    min_lon + lon_step, min_lat + lat_step
])

print(f"\nTesting single chunk download...")
import numpy as np
chunk_img = planet_img.select(['B1']).clip(chunk_geom).unmask(0)
chunk_sample = chunk_img.sampleRectangle(region=chunk_geom, defaultValue=0)
chunk_data = chunk_sample.getInfo()

if chunk_data and 'properties' in chunk_data:
    arr = np.array(chunk_data['properties']['B1'])
    print(f"Chunk shape: {arr.shape}")
    print(f"Chunk mean: {arr.mean()}")
    print(f"Nonzero: {(arr > 0).sum()} / {arr.size}")
