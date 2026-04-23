"""Test parallel chunked download - simplified version"""
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
from Auth.AuthGEE import find_pairs, regions, PLANET_COLLECTION_ID
from chunked_download_fast import download_image_chunked_parallel
import time

region = regions[0]
print(f"\n=== Fast Chunked Download Test ===")
print(f"Region: {region['name']}")
print(f"Using cached pairs (or will compute if needed)...\n")

pairs = find_pairs(region['aoi'], region['start'], region['end'], PLANET_COLLECTION_ID)
planet_collection = pairs.get('planet_collection')
matched = pairs.get('matched_pairs', [])

if not matched:
    print("❌ No matched pairs found!")
else:
    first = matched[0]
    print(f"✓ Testing Planet image from: {first['planet_date']}\n")
    
    # Get the mosaicked image
    planet_img = planet_collection.filter(ee.Filter.eq('id_pref', first['planet_id'])).first()
    bands = ['B1', 'B2', 'B3', 'B4']
    
    # Download with parallel chunking
    print(f"Downloading with 9x9 grid (81 chunks), 6 parallel workers...")
    start = time.time()
    data = download_image_chunked_parallel(
        planet_img, 
        region['aoi'], 
        bands, 
        scale=3, 
        chunk_grid_size=9,  # 9x9 = 81 smaller chunks to stay under 262k pixel limit
        max_workers=6
    )
    elapsed = time.time() - start
    
    if data is not None:
        print(f"\n✓ SUCCESS in {elapsed:.1f}s")
        print(f"  Shape: {data.shape}")
        print(f"  Mean values: B1={data[:,:,0].mean():.1f}, B2={data[:,:,1].mean():.1f}, B3={data[:,:,2].mean():.1f}, B4={data[:,:,3].mean():.1f}")
        nonzero = (data.sum(axis=2) > 0).sum()
        total = data.shape[0] * data.shape[1]
        print(f"  Valid pixels: {nonzero}/{total} ({100*nonzero/total:.1f}%)")
        print(f"\n🚀 Parallel chunking works! Much faster than 81 sequential chunks.")
    else:
        print(f"\n❌ Download failed")
