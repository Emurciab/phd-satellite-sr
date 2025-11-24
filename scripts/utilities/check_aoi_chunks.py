"""
Check AOI size and calculate optimal chunking strategy
"""
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import ee
from Auth.AuthGEE import regions

def check_aoi_size():
    region = regions[0]
    aoi = region['aoi']
    
    # Get bounds
    bounds = aoi.bounds().getInfo()
    coords = bounds['coordinates'][0]
    
    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    min_lat = min(c[1] for c in coords)
    max_lat = max(c[1] for c in coords)
    
    print(f"Region: {region['name']}")
    print(f"Bounds: {min_lon:.4f}, {min_lat:.4f} to {max_lon:.4f}, {max_lat:.4f}")
    
    # Calculate size in meters (approximate)
    import math
    lat_m = (max_lat - min_lat) * 111320  # 1 degree lat ≈ 111.32 km
    lon_m = (max_lon - min_lon) * 111320 * math.cos(math.radians((max_lat + min_lat) / 2))
    
    print(f"Size: ~{lat_m:.0f}m x {lon_m:.0f}m")
    
    # At 3m resolution
    pixels_3m = (lat_m / 3) * (lon_m / 3)
    print(f"Pixels at 3m: ~{pixels_3m:,.0f}")
    print(f"Bands: 4")
    print(f"Bytes (float32): ~{pixels_3m * 4 * 4 / 1024**2:.1f} MB")
    
    # getDownloadURL limit: 50MB
    # sampleRectangle limit: 262,144 pixels
    
    print(f"\nLimits:")
    print(f"  getDownloadURL: 50 MB max")
    print(f"  sampleRectangle: 262,144 pixels max (~512x512)")
    
    # Calculate how many chunks we need
    max_pixels_per_chunk = 262144  # sampleRectangle limit
    num_chunks = math.ceil(pixels_3m / max_pixels_per_chunk)
    
    # Calculate chunk grid (square chunks)
    chunks_per_side = math.ceil(math.sqrt(num_chunks))
    
    print(f"\nSolution: Split into ~{chunks_per_side}x{chunks_per_side} = {chunks_per_side**2} chunks")
    print(f"Each chunk: ~{lat_m/chunks_per_side:.0f}m x {lon_m/chunks_per_side:.0f}m")
    print(f"Each chunk: ~{pixels_3m/chunks_per_side**2:,.0f} pixels")

if __name__ == '__main__':
    check_aoi_size()
