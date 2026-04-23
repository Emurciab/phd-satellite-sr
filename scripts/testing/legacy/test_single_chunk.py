"""Test single chunk download speed"""
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
import pickle
import time
import numpy as np
from Auth.AuthGEE import regions

# Load cached pairs
region = regions[0]
with open('test_pairs_cache.pkl', 'rb') as f:
    pairs = pickle.load(f)

planet_collection = pairs['planet_collection']
matched = pairs['matched_pairs']
first = matched[0]

planet_img = planet_collection.filter(ee.Filter.eq('id_pref', first['planet_id'])).first()
bands = ['B1', 'B2', 'B3', 'B4']

# Get one chunk geometry (center chunk of 9x9 grid)
aoi = region['aoi']
bounds = aoi.bounds().getInfo()['coordinates'][0]
lons = [p[0] for p in bounds]
lats = [p[1] for p in bounds]
min_lon, max_lon = min(lons), max(lons)
min_lat, max_lat = min(lats), max(lats)

lon_step = (max_lon - min_lon) / 9
lat_step = (max_lat - min_lat) / 9

# Center chunk (i=4, j=4)
chunk_min_lon = min_lon + 4 * lon_step
chunk_max_lon = min_lon + 5 * lon_step
chunk_min_lat = max_lat - 5 * lat_step
chunk_max_lat = max_lat - 4 * lat_step

chunk_geom = ee.Geometry.Rectangle([chunk_min_lon, chunk_min_lat, chunk_max_lon, chunk_max_lat])

print(f"Testing single chunk download...")
print(f"Image: {first['planet_date']}")
print(f"Chunk: center of 9x9 grid\n")

# Download chunk
start = time.time()
try:
    chunk_img = planet_img.select(bands).clip(chunk_geom).unmask(0)
    chunk_img = chunk_img.reproject(crs='EPSG:4326', scale=3)
    chunk_sample = chunk_img.sampleRectangle(region=chunk_geom, defaultValue=0)
    chunk_data = chunk_sample.getInfo()
    elapsed = time.time() - start
    
    if chunk_data and 'properties' in chunk_data:
        props = chunk_data['properties']
        band_arrays = [np.array(props[b], dtype=np.float32) for b in bands]
        chunk_array = np.stack(band_arrays, axis=-1)
        
        print(f"✅ Success in {elapsed:.2f}s")
        print(f"   Shape: {chunk_array.shape}")
        print(f"   Pixels: {chunk_array.shape[0] * chunk_array.shape[1]:,}")
        print(f"   Mean: {chunk_array.mean():.1f}")
        print(f"   Nonzero: {(chunk_array.sum(axis=2) > 0).sum()}")
        
        print(f"\n📊 Extrapolated for 81 chunks:")
        print(f"   Sequential: {81 * elapsed:.1f}s ({81 * elapsed / 60:.1f} min)")
        print(f"   6 parallel: ~{81 * elapsed / 6:.1f}s ({81 * elapsed / 360:.1f} min)")
    else:
        print(f"❌ No data returned")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Error after {elapsed:.2f}s: {e}")
