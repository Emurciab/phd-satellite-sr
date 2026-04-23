"""Test just the download part without find_pairs overhead"""
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
import pickle
from Auth.AuthGEE import regions
from chunked_download_fast import download_image_chunked_parallel
import time

# Load cached pairs to skip find_pairs()
region = regions[0]
cache_file = 'test_pairs_cache.pkl'

if os.path.exists(cache_file):
    print("Loading cached pairs...")
    with open(cache_file, 'rb') as f:
        pairs = pickle.load(f)
else:
    print("Computing pairs (will cache for next run)...")
    from Auth.AuthGEE import find_pairs, PLANET_COLLECTION_ID
    pairs = find_pairs(region['aoi'], region['start'], region['end'], PLANET_COLLECTION_ID)
    with open(cache_file, 'wb') as f:
        pickle.dump(pairs, f)

planet_collection = pairs['planet_collection']
matched = pairs['matched_pairs']
first = matched[0]

print(f"\n=== Download Speed Test ===")
print(f"Image: {first['planet_date']}")
print(f"Config: 9x9 grid (81 chunks), 6 parallel workers\n")

planet_img = planet_collection.filter(ee.Filter.eq('id_pref', first['planet_id'])).first()
bands = ['B1', 'B2', 'B3', 'B4']

start = time.time()
data = download_image_chunked_parallel(
    planet_img, 
    region['aoi'], 
    bands, 
    scale=3, 
    chunk_grid_size=9,
    max_workers=6
)
elapsed = time.time() - start

if data is not None and data.sum() > 0:
    print(f"\n✅ SUCCESS in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Shape: {data.shape}")
    nonzero = (data.sum(axis=2) > 0).sum()
    total = data.shape[0] * data.shape[1]
    print(f"   Valid pixels: {nonzero}/{total} ({100*nonzero/total:.1f}%)")
    print(f"   Mean values: B1={data[:,:,0].mean():.1f}, B2={data[:,:,1].mean():.1f}, B3={data[:,:,2].mean():.1f}, B4={data[:,:,3].mean():.1f}")
    print(f"\n   Speed: {elapsed/81:.2f}s per chunk")
    print(f"   Throughput: {total/elapsed:.0f} pixels/second")
else:
    print(f"\n❌ Failed or all zeros")
