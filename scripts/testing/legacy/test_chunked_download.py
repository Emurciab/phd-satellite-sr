"""
Test chunked download approach
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
from Auth.AuthGEE import find_pairs, regions, PLANET_COLLECTION_ID
from chunked_download import download_image_chunked

def test():
    region = regions[0]
    print(f"Testing chunked download for: {region['name']}\n")
    
    # Get pairs
    pairs = find_pairs(region['aoi'], region['start'], region['end'], PLANET_COLLECTION_ID)
    matched = pairs.get('matched_pairs', [])
    planet_collection = pairs.get('planet_collection')
    
    if not matched:
        print("❌ No matched pairs")
        return
    
    # Test first pair
    pair = matched[0]
    print(f"Downloading Planet image: {pair['planet_date']}")
    
    planet_img = planet_collection.filter(ee.Filter.eq('id_pref', pair['planet_id'])).first()
    
    # Check what bands and properties the image has
    print(f"   Image properties: {planet_img.getInfo()['properties']}")
    print(f"   Image bands: {planet_img.bandNames().getInfo()}")
    
    planet_bands = ['B1', 'B2', 'B3', 'B4']
    
    # Download using chunks
    hr_array = download_image_chunked(planet_img, region['aoi'], planet_bands, scale=3, chunk_grid_size=9)
    
    if hr_array is not None:
        print(f"\n✅ SUCCESS!")
        print(f"   Shape: {hr_array.shape}")
        print(f"   Stats: min={hr_array.min():.1f}, max={hr_array.max():.1f}, mean={hr_array.mean():.1f}")
        print(f"   Sum: {hr_array.sum():.1f}")
        
        if hr_array.sum() == 0:
            print(f"   ⚠️  All zeros!")
        else:
            print(f"   🎉 Has real Planet data!")
            
        # Show sample
        print(f"\n   Sample (top-left 5x5, band 0):")
        print(hr_array[:5, :5, 0])
    else:
        print(f"\n❌ FAILED")

if __name__ == '__main__':
    test()
