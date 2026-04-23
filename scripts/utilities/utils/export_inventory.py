#!/usr/bin/env python3
"""
Export S2 and PlanetScope image inventory to CSV/Excel files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from Auth.AuthGEE import find_pairs, export_image_inventory, PLANET_COLLECTION_ID, regions

def main():
    # Get pairs for STA-2 region
    region = regions[0]
    print(f'Processing region: {region["name"]}')
    print(f'Date range: {region["start"]} to {region["end"]}')
    
    print("\n🔍 Finding image pairs...")
    pair_results = find_pairs(
        region['aoi'], 
        region['start'], 
        region['end'], 
        PLANET_COLLECTION_ID
    )
    
    print(f'\n📊 Results:')
    print(f'   S2 images: {len(pair_results.get("s2_all_dates", []))}')
    print(f'   Planet images: {len(pair_results.get("planet_all_dates", []))}')
    print(f'   Matched pairs: {len(pair_results["matched_pairs"])}')
    
    # Export inventory to files (without metadata for speed)
    print("\n💾 Exporting inventory files...")
    files = export_image_inventory(
        pair_results,
        output_path='./data/inventory/sta2',
        include_metadata=False,  # Set to True for full metadata (slower)
        region_name=region['name']
    )
    
    print(f'\n✅ Export complete!')
    for name, path in files.items():
        print(f'   {name}: {path}')


if __name__ == '__main__':
    main()
