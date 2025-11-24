"""Quick check: Does Planet collection have any valid pixels?"""
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
from Auth.AuthGEE import find_pairs, regions, PLANET_COLLECTION_ID

region = regions[0]
print(f"Checking {region['name']}")

pairs = find_pairs(region['aoi'], region['start'], region['end'], PLANET_COLLECTION_ID)
planet_collection = pairs.get('planet_collection')
matched = pairs.get('matched_pairs', [])

if matched:
    first = matched[0]
    print(f"\nTesting first Planet image: {first['planet_date']}")
    
    # Get the image
    planet_img = planet_collection.filter(ee.Filter.eq('id_pref', first['planet_id'])).first()
    
    # Check if image has ANY unmasked pixels
    print("\n1. Checking pixel count...")
    pixel_count = planet_img.select('B1').reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region['aoi'],
        scale=10,
        maxPixels=1e9
    ).getInfo()
    
    print(f"   Unmasked pixels: {pixel_count}")
    
    # Check mean values
    print("\n2. Checking mean values...")
    stats = planet_img.select(['B1','B2','B3','B4']).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region['aoi'],
        scale=50,
        maxPixels=1e9
    ).getInfo()
    
    print(f"   Mean values: {stats}")
    
    # Try sampling a small point
    print("\n3. Trying point sample at AOI centroid...")
    centroid = region['aoi'].centroid()
    sample = planet_img.select(['B1','B2','B3','B4']).sample(
        region=centroid.buffer(100),
        scale=10,
        numPixels=1
    ).first().getInfo()
    
    if sample and 'properties' in sample:
        props = sample['properties']
        print(f"   Sample values: B1={props.get('B1')}, B2={props.get('B2')}, B3={props.get('B3')}, B4={props.get('B4')}")
    else:
        print(f"   Sample returned: {sample}")
