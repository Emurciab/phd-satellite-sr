"""
Minimal test to verify getDownloadURL approach works for Planet mosaics
"""
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
import numpy as np
import requests
import io
from Auth.AuthGEE import find_pairs, regions, PLANET_COLLECTION_ID

def test_download():
    print("🧪 Testing getDownloadURL download logic...")
    
    region = regions[0]
    print(f"Region: {region['name']}")
    
    # Get pairs
    print("Finding pairs...")
    pairs = find_pairs(region['aoi'], region['start'], region['end'], PLANET_COLLECTION_ID)
    matched = pairs.get('matched_pairs', [])
    planet_collection = pairs.get('planet_collection')
    
    if not matched:
        print("❌ No matched pairs found")
        return
    
    # Test first pair
    pair = matched[0]
    print(f"\n📥 Testing download for: {pair['planet_date']}")
    print(f"   Planet ID: {pair['planet_id']}")
    
    # Get image from collection
    planet_img = planet_collection.filter(ee.Filter.eq('id_pref', pair['planet_id'])).first()
    
    # Check bands
    print("   Checking bands...", end=' ', flush=True)
    available_bands = planet_img.bandNames().getInfo()
    print(f"✓ {len(available_bands)} bands")
    
    # Use 4-band for simplicity
    planet_bands = ['B1', 'B2', 'B3', 'B4']
    
    # Download via getDownloadURL with unmask
    print("   Downloading via getDownloadURL...", end=' ', flush=True)
    
    try:
        planet_selected = planet_img.select(planet_bands).clip(region['aoi'])
        planet_unmasked = planet_selected.unmask(0)
        
        planet_url = planet_unmasked.getDownloadURL({
            'scale': 3,  # Planet resolution
            'region': region['aoi'],
            'format': 'NPY'
        })
        
        response = requests.get(planet_url)
        hr_data = np.load(io.BytesIO(response.content), allow_pickle=True)
        hr_full = hr_data.copy()
        
        print(f"✓")
        
        # Check the data
        print(f"\n✅ SUCCESS!")
        print(f"   Shape: {hr_full.shape}")
        print(f"   Stats: min={hr_full.min():.1f}, max={hr_full.max():.1f}, mean={hr_full.mean():.1f}")
        print(f"   Sum: {hr_full.sum():.1f}")
        
        if hr_full.sum() == 0:
            print(f"   ⚠️  WARNING: All zeros!")
        else:
            print(f"   🎉 Has real Planet data!")
            
        # Sample some pixel values
        print(f"\n   Sample pixel values:")
        print(f"   Top-left corner (5x5, band 0):")
        print(f"   {hr_full[:5, :5, 0]}")
        
    except Exception as e:
        print(f"❌ FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_download()
