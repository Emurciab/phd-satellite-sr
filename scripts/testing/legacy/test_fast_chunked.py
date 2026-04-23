"""
Test fast parallel chunked download
"""
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
import sys
sys.path.insert(0, '/media/eamurciab/TOSHIBA_EXT1/PhD/SR/Auth')
# Import AuthGEE to trigger ee.Initialize()
from AuthGEE import find_pairs, regions, PLANET_COLLECTION_ID
sys.path.insert(0, '/media/eamurciab/TOSHIBA_EXT1/PhD/SR')
from chunked_download_fast import download_image_chunked_parallel

region = regions[0]
aoi = region['aoi']

print(f"\n=== Testing Fast Chunked Download ===")
print(f"Region: {region['name']}")
print(f"Method: 3x3 grid with 6 parallel workers\n")

# Get properly processed Planet collection
print(f"Finding matched pairs...")
pairs = find_pairs(aoi, region['start'], region['end'], PLANET_COLLECTION_ID)
planet_collection = pairs.get('planet_collection')
matched = pairs.get('matched_pairs', [])

if not matched:
    print("No matched pairs found!")
    sys.exit(1)

first = matched[0]
print(f"Using Planet image from: {first['planet_date']}")

# Get the image
planet_img = planet_collection.filter(ee.Filter.eq('id_pref', first['planet_id'])).first()
bands = ['B1', 'B2', 'B3', 'B4']

print(f"Downloading Planet image...")
import time
start = time.time()
data = download_image_chunked_parallel(planet_img, aoi, bands, scale=3, chunk_grid_size=3, max_workers=6)
elapsed = time.time() - start

if data is not None:
    print(f"\n✓ Success in {elapsed:.1f}s")
    print(f"  Shape: {data.shape}")
    print(f"  Mean values: B1={data[:,:,0].mean():.1f}, B2={data[:,:,1].mean():.1f}, B3={data[:,:,2].mean():.1f}, B4={data[:,:,3].mean():.1f}")
    print(f"  Nonzero pixels: {(data.sum(axis=2) > 0).sum()}")
else:
    print(f"\n❌ Failed!")
