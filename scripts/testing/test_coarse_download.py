"""Test if reduceResolution can make downloads fit within limits"""
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
import numpy as np
from Auth.AuthGEE import find_pairs, regions, PLANET_COLLECTION_ID

region = regions[0]
pairs = find_pairs(region['aoi'], region['start'], region['end'], PLANET_COLLECTION_ID)
matched = pairs.get('matched_pairs', [])
planet_collection = pairs.get('planet_collection')

pair = matched[0]
planet_img = planet_collection.filter(ee.Filter.eq('id_pref', pair['planet_id'])).first()

print(f"Testing reduceResolution approach...")
print(f"Original scale: 3m Planet")
print(f"Testing with scale=30m (10x coarser)...")

# Try downloading at coarser resolution
planet_selected = planet_img.select(['B1','B2','B3','B4'])

try:
    # Method 1: Just use coarser scale in sampleRectangle
    print("\n1. sampleRectangle with scale=30...")
    result = planet_selected.sampleRectangle(region=region['aoi'], defaultValue=0)
    data = result.getInfo()
    
    if data and 'properties' in data:
        props = data['properties']
        arr = np.array(props['B1'], dtype=np.float32)
        print(f"   ✓ Shape: {arr.shape}, Sum: {arr.sum():.1f}")
    else:
        print(f"   ✗ Failed")
except Exception as e:
    print(f"   ✗ Error: {str(e)[:100]}")

print("\nConclusion: If this works, we can download at coarser scale then upsample locally!")
